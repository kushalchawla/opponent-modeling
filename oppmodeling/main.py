from argparse import ArgumentParser
from os.path import join
from os import environ

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from oppmodeling.basebuilder import add_builder_specific_args
from oppmodeling.utils import get_run_dir

from oppmodeling.dm import BaseDM

#set anomaly detection
torch.autograd.set_detect_anomaly(True)

def main(args):
    
    print("In main: ")
    print(vars(args))
    """ main """

    pl.seed_everything(args.overall_seed)

    # ------------------------------------------------------------------
    # Data
    print("ARGS:", args)
    args.save_dir = args.default_root_dir #for convenience
    
    if(args.tokenizer):
        print(f"using already available tokenizer from {args.tokenizer}")
        tok = torch.load(args.tokenizer)
        dm = BaseDM(args, tokenizer=tok)
    else:
        print("No supplied tokenizer. Recreating the tokenizer.")
        dm = BaseDM(args)
    print("DataLoader")
    print("Batch size: ", args.batch_size)
    print("num workers: ", args.num_workers)
    print("tokenizer path: ", args.tokenizer)
    print("speaker indices: ", dm.speaker_indices)
    # ------------------------------------------------------------------
    # Checkpoint callback (early stopping)
    checkpoint_callback = None
    callbacks = None
    local_rank = environ.get("LOCAL_RANK", 0)
    if local_rank == 0:
        print("LOCAL_RANK: ", local_rank)
        print("Logging -> ", args.save_dir)

        name = args.model
        desc = f"{name} training"
        logger = TensorBoardLogger(args.save_dir, name=name)
        ch_path = join(logger.log_dir, "checkpoints")
        print("ch_path", ch_path)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=ch_path,
            filename="{epoch}-{val_loss:.5f}-{val_acc:.5f}",
            save_top_k=args.top_k_checkpoints,
            save_last=False,
            mode="max",
            monitor="val_acc",
            verbose=True,
            save_on_train_epoch_end=True
        )

        # Save the used tokenizer
        tokenizer_path = join(logger.experiment.log_dir, "tokenizer.pt")
        torch.save(dm.tokenizer, tokenizer_path)
        print("tokenizer saved -> ", tokenizer_path)

        if args.early_stopping:
            print(f"Early stopping (patience={args.patience})")
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                strict=True,  # crash if "monitor" is not found in val metrics
                verbose=True,
            )
            callbacks = [early_stop_callback]
        print("-" * 50)

    # ------------------------------------------------------------------
    # Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        #checkpoint_callback=checkpoint_callback,
        callbacks=[checkpoint_callback],
    )

    # ------------------------------------------------------------------
    # Data
    dm.prepare_data()
    dm.setup("fit")
    
    # ------------------------------------------------------------------
    # Model
    if args.model == "hierarchical":
        from oppmodeling.models.hierarchical import HierarchicalFramework

        model = HierarchicalFramework(
            speaker_indices=dm.speaker_indices,
            n_vocab=len(dm.tokenizer),
            pad_idx=dm.tokenizer.pad_token_id,
            **vars(args),
        )
    
    print("\n-----", "Model", "-----")
    print("pad_idx: ", model.pad_idx)
    print("n_vocab: ", len(dm.tokenizer))
    if "n_layer" in args:
        print("n_layer: ", args.n_layer)
    else:
        print("n_layer: ", model.n_layer)

    if "n_head" in args:
        print("n_head: ", args.n_head)
    else:
        print("n_head: ", model.n_head)
    if "n_embd" in args:
        print("n_embd: ", args.n_embd)
    else:
        print("n_embd: ", model.n_embd)
    print()

    # ------------------------------------------------------------------
    # FIT
    trainer.fit(model, datamodule=dm)
    

if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaseDM.add_data_specific_args(parser)
    parser.add_argument("--early_stopping", default=False, type=bool)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument(
        "--model",
        type=str,
        default="hierarchical",  # sparse, hugging
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["combined"],
    )
    parser.add_argument(
        "--cv_no",
        type=int,
    )
    parser.add_argument(
        "--overall_seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--top_k_checkpoints",
        type=int,
        default=1,
    )

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # Choose Model
    if temp_args.model == "hierarchical":
        from oppmodeling.models.hierarchical import HierarchicalFramework

        parser = HierarchicalFramework.add_model_specific_args(parser)
    else:
        raise NotImplementedError

    # Add all datasets
    datasets = temp_args.datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()

    # Where to save the training
    print()
    print("-" * 70)
    args.save_dir = get_run_dir(__file__)
    print(args.save_dir)
    print()

    main(args)