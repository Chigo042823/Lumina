use std::time::Instant;

use crate::{data::{ClassificationBatch, ClassificationBatcher}, dataset::SBW8Loader, model::Lumina};

use burn::{config::Config, data::{dataloader::DataLoaderBuilder, dataset::vision::ImageFolderDataset}, module::Module, nn::loss::CrossEntropyLossConfig, optim::{decay::WeightDecayConfig, AdamConfig}, prelude::Backend, record::{CompactRecorder, Recorder}, tensor::{backend::AutodiffBackend, Int, Tensor}, train::{metric::{AccuracyMetric, CpuMemory, CpuUse, LossMetric, PrecisionMetric, RecallMetric}, renderer, ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep}};

const ARTIFACT_DIR: &str = "tmp";

impl<B: Backend> Lumina<B> {
    pub fn forward_classification(&self, inputs: Tensor<B, 4>, targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
        let x = self.forward(inputs);
        let loss = CrossEntropyLossConfig::new()
            .init(&x.device())
            .forward(x.clone(), targets.clone());

        ClassificationOutput { loss, output: x, targets}
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, ClassificationOutput<B>> for Lumina<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, ClassificationOutput<B>> for Lumina<B> {
    fn step(&self, batch: ClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 40)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.001)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {
    // let record: crate::model::LuminaRecord<B> = CompactRecorder::new()
    //     .load(format!("{ARTIFACT_DIR}/model").into(), &device)
    //     .expect("Error loading model");
    
    create_artifact_dir(ARTIFACT_DIR);

    config
        .save(format!("{ARTIFACT_DIR}/config.json"))
        .expect("Config should be saved successfully");

    // let model = Lumina::<B>::new(&device).load_record(record);
    let model = Lumina::<B>::new(&device);

    B::seed(config.seed);

    let batcher_train = ClassificationBatcher::new();
    let batcher_valid = ClassificationBatcher::new();
    
    let dl_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(ImageFolderDataset::sbw8("train"));

    let dl_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(ImageFolderDataset::sbw8("test"));
    
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train(CpuMemory::new())
        .metric_train(CpuUse::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
                model, 
            config.optimizer
                .with_weight_decay(Some(WeightDecayConfig::new(5e-5)))
                .init(), 
            config.learning_rate
        );
    
    let now = Instant::now();
    let model_trained = learner.fit(dl_train, dl_test);
    let finished = now.elapsed().as_secs();
   
    println!("Training completed in {}m {}s", finished/60, finished % 60);
    println!("\n");
    
    model_trained  
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Error saving model.");

}
