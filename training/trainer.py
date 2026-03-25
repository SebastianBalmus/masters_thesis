from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from trl import SFTTrainer


class CurriculumSFTTrainer(SFTTrainer):
    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func,
        dataset_name,
    ):
        # If dataset is already a tokenized PyTorch IterableDataset, bypass TRL preprocessing.
        if isinstance(dataset, TorchIterableDataset):
            sample = next(iter(dataset))
            required_keys = {"input_ids", "attention_mask", "labels"}

            if not required_keys.issubset(sample.keys()):
                raise ValueError(
                    f"{dataset_name} is an IterableDataset but missing required keys "
                    f"{required_keys}. Found keys: {list(sample.keys())}"
                )

            return dataset

        return super()._prepare_dataset(
            dataset=dataset,
            processing_class=processing_class,
            args=args,
            packing=packing,
            formatting_func=formatting_func,
            dataset_name=dataset_name,
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )
