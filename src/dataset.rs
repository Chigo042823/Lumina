use std::{path::PathBuf, process::exit};

use burn::data::dataset::{transform::{PartialDataset, ShuffledDataset}, vision::{ImageDatasetItem, ImageFolderDataset}, Dataset};

const ROOT: &str = "C:/sbw7";

type D = PartialDataset<ShuffledDataset<ImageFolderDataset, ImageDatasetItem>, ImageDatasetItem>;

pub trait SBW8Loader {
    fn sbw8(split: &str) -> D;
}

impl SBW8Loader for ImageFolderDataset {
    fn sbw8(split: &str) -> D {
        let dataset = Self::new_classification(ROOT)
            .expect("Error loading SBW8 training samples!");

        let len = dataset.len();

        let shuffled = ShuffledDataset::with_seed(dataset, 42);
        match split {
            "train" => PartialDataset::new(shuffled, 0, len * 8 / 10),
            "test" => PartialDataset::new(shuffled, len * 8 / 10, len),
            _ => {
                println!("Invalid split type");
                exit(0);  
            }
        }  
    }
}   