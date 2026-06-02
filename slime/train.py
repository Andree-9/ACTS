import logging
import re
import shutil
from pathlib import Path

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _checkpoint_rollout_id(path: Path) -> int | None:
    match = re.fullmatch(r"iter_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _rollout_state_id(path: Path) -> int | None:
    match = re.fullmatch(r"global_dataset_state_dict_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def prune_old_checkpoints(args) -> None:
    keep = getattr(args, "max_checkpoints_to_keep", None)
    if keep is None:
        return
    if keep <= 0:
        logger.warning("Skip checkpoint pruning because max_checkpoints_to_keep=%s", keep)
        return
    if getattr(args, "async_save", False):
        logger.warning("Skip checkpoint pruning when async_save is enabled.")
        return
    if not args.save:
        return

    save_dir = Path(args.save)
    if not save_dir.exists():
        return

    checkpoints: list[tuple[int, Path]] = []
    for child in save_dir.iterdir():
        if not child.is_dir():
            continue
        rollout_id = _checkpoint_rollout_id(child)
        if rollout_id is not None:
            checkpoints.append((rollout_id, child))

    checkpoints.sort(key=lambda item: item[0])
    if len(checkpoints) <= keep:
        return

    kept_ids = {rollout_id for rollout_id, _ in checkpoints[-keep:]}
    for rollout_id, path in checkpoints[:-keep]:
        logger.info("Prune old checkpoint %s", path)
        shutil.rmtree(path)

    rollout_dir = save_dir / "rollout"
    if not rollout_dir.is_dir():
        return
    for state_path in rollout_dir.glob("global_dataset_state_dict_*.pt"):
        rollout_id = _rollout_state_id(state_path)
        if rollout_id is not None and rollout_id not in kept_ids:
            logger.info("Prune old rollout dataset state %s", state_path)
            state_path.unlink()


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))
        prune_old_checkpoints(args)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout=args.num_rollout):
            save(rollout_id)

        offload_train(rollout_id)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        if not args.critic_train_only:
            actor_model.update_weights()
        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
