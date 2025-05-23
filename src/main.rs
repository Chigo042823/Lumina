#![recursion_limit = "256"]

use burn::{
    backend::{
        libtorch::{
            LibTorch,
            LibTorchDevice
        }, Autodiff
    }, data::{
        dataloader::batcher::Batcher, dataset::{vision::ImageFolderDataset, Dataset}
    }, module::Module, optim::AdamConfig, prelude::Backend, record::CompactRecorder, tensor::{self, activation::softmax, check_closeness, Tensor, TensorData} 
};

use image::open;
use lumina::{data::ClassificationBatcher, dataset::SBW8Loader, model::Lumina, training::{train, TrainingConfig}};

type B = LibTorch<f32, i8>;
type ADB = Autodiff<B>;

fn main() {
    let device = LibTorchDevice::default();

    // train::<ADB>(
    //     TrainingConfig::new(AdamConfig::new()), device);

    let dataset = ImageFolderDataset::sbw8("train");

    let item = dataset.get(7).unwrap();

    let path = item.image_path.clone();

    println!("{:?}", &path);

    let batcher = ClassificationBatcher::new();

    let t_data: lumina::data::ClassificationBatch<B> = batcher.batch(vec![item.clone()], &device);

    let t_raw = t_data.images;

    let img = open(path).unwrap();

    let i_buf = img.into_rgb8().to_vec();

    let i_t = TensorData::new(i_buf, [1, 224, 224, 3]);

    let tensor: Tensor<B, 4> = Tensor::from_data(i_t.convert::<<LibTorch as Backend>::FloatElem>(), &device)
        .swap_dims(3, 2)
        .swap_dims(2, 1);
    let input = tensor / 255;

    // println!("{}", input);

    // println!("{}", t_raw);

    // check_closeness(&input,&t_raw);

    let model: Lumina<B> = Lumina::new(&device);

    let recorder = CompactRecorder::new();

    let model = model.load_file("tmp/checkpoint/model-40.mpk", &recorder, &device).unwrap();

    let pred = model.forward(t_raw);

    println!("{:?}", item.annotation);
    println!("{}", softmax(pred, 1));
    

    // Lumina::<B>::new(&WgpuDevice::default()).print();

    // infer::<B>("D:/lumina/tmp", NdArrayDevice::Cpu, sample);

}