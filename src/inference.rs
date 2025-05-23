use std::fmt::format;

use burn::{data::{dataloader::batcher::Batcher, dataset::vision::{Annotation, ImageDatasetItem}}, module::Module, prelude::Backend, record::{CompactRecorder, Recorder}, tensor::Tensor};

use crate::{data::ClassificationBatcher, model::Lumina};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: ImageDatasetItem) {
    let record: crate::model::LuminaRecord<B> = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Error loading model");

    let model = Lumina::<B>::new(&device);

    let mut label = 0;
    if let Annotation::Label(l) = item.annotation {
        label = l;
    };

    let batcher = ClassificationBatcher::new();
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let index = output.clone().argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("{} - {}% -- Expected: {}", index, output, label);
}