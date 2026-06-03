import random

from accelerate.state import PartialState
from rewards import outcome_reward
from transformers.trainer_utils import get_last_checkpoint, set_seed
from trl import ModelConfig, get_peft_config
from utils.arguments import H4ArgumentParser, ScriptArguments
from utils.logging_utils import setup_logger
from utils.train_utils import get_datasets, load_model_and_tokenizer

from train.grail_trainer import GrailConfigV2, GrailTrainerV2

REWARD_FUNCS = [outcome_reward]
REWARD_WEIGHTS = [1]


def main():
    parser = H4ArgumentParser((ScriptArguments, ModelConfig, GrailConfigV2))
    script_args, model_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logger = setup_logger(training_args, script_args)

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ########################################
    # Model & Tokenizer & Reward functions
    ########################################
    logger.info("*** Loading pretrained model and tokenizer ***")

    model, model_kwargs, tokenizer = load_model_and_tokenizer(
        script_args, model_args, training_args
    )

    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***")

    raw_datasets = get_datasets(
        script_args,
        splits=script_args.dataset_splits,
        configs=script_args.dataset_configs,
        columns_to_keep=None,
    )

    def add_boxed_prompt(sample):
        prompt = sample["prompt"][0]["content"]
        sample["prompt"][0]["content"] = (
            prompt
            + " Please reason step by step, and put your final answer within \boxed{}."
        )
        return sample

    with PartialState().main_process_first():
        raw_datasets = raw_datasets.map(
            add_boxed_prompt,
            num_proc=script_args.preprocessing_num_workers,
            desc="Adding boxed output hint",
        )

    # split the dataset into train and test if requires evaluation
    if training_args.do_eval and "test" not in raw_datasets:
        raw_datasets = raw_datasets.train_test_split(test_size=0.05)

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    if training_args.debug:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(30))

    if PartialState().is_main_process:
        print(raw_datasets)
        for index in random.sample(range(len(raw_datasets["train"])), 2):
            logger.info(
                f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}"
            )

    train_dataset = raw_datasets.get("train", None)
    eval_dataset = raw_datasets.get("test", None)

    ################
    # Instantiate GRPO trainer
    ################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)

    training_args.reward_weights = REWARD_WEIGHTS

    # trainer = CustomGRPOTrainer(
    trainer = GrailTrainerV2(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = (
            get_last_checkpoint(training_args.output_dir)
            if isinstance(training_args.resume_from_checkpoint, bool)
            else training_args.resume_from_checkpoint
        )
        if checkpoint is not None:
            logger.warning(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(
                f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch"
            )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Training complete ***")

    ##################################
    # Save model
    ##################################
    logger.info("*** Saving model ***")
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
