import argparse
from pathlib import Path

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd

from chemprop import data, featurizers, models, nn

def main():
    parser = argparse.ArgumentParser(description="Train a Chemprop model for molecular property prediction.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file containing SMILES and targets.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory path to save trained model checkpoints.")
    # parser.add_argument("--smiles_column", type=str, default="smiles", help="Name of the column containing SMILES strings.")
    # parser.add_argument("--target_columns", type=str, nargs="+", required=True, help="Names of the target columns.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    args = parser.parse_args()

    # Load data
    input_path = Path(args.input)
    df_input = pd.read_csv(input_path, sep=";")

    # smis = df_input.loc[:, args.smiles_column].values
    # ys = df_input.loc[:, args.target_columns].values
    smis = df_input.loc[:, "SMILES_acc"].values
    targets_df = df_input.filter(regex="^acc").select_dtypes("number")
    targets_df = targets_df.loc[:, (targets_df.var() > 0) & (targets_df.max() < 1e10)].dropna()
    smis = smis[targets_df.index]
    ys = targets_df.values

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    # Split data
    mols = [d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    # Featurization and Datasets
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    val_dset.normalize_targets(scaler)

    test_dset = data.MoleculeDataset(test_data[0], featurizer)

    # DataLoaders
    train_loader = data.build_dataloader(train_dset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    val_loader = data.build_dataloader(val_dset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    test_loader = data.build_dataloader(test_dset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # MPNN Components
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    # ffn = nn.RegressionFFN(output_transform=output_transform)
    ffn = nn.RegressionFFN(output_transform=output_transform, n_tasks=ys.shape[1])
    
    batch_norm = True
    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Trainer Setup
    checkpointing = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="best-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpointing],
    )

    # Train and test
    trainer.fit(mpnn, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, weights_only=False)

if __name__ == "__main__":
    main()
