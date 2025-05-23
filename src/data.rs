use burn::{data::{dataloader::batcher::Batcher, dataset::vision::{Annotation, ImageDatasetItem, PixelDepth}}, prelude::Backend, tensor::{Device, ElementConversion, Int, Shape, Tensor, TensorData}};

const IMG_SIZE: usize = 224;
const CHANNELS: usize = 3;

pub struct Normalizer<B: Backend> {
    pub mean: Tensor<B, 4>,
    pub std: Tensor<B, 4>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &Device<B>) -> Self {
        let mean = Tensor::zeros(Shape::new([1, 3, 1, 1]), device);
        let std = Tensor::zeros(Shape::new([1, 3, 1, 1]), device);
        Self { mean, std }
    }

    pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        (input - self.mean.clone()) / self.std.clone()
    }
}

#[derive(Clone)]
pub struct ClassificationBatcher {}

impl ClassificationBatcher {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
    pub image_paths: Vec<String>
}

impl<B: Backend> Batcher<B, ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher {
    fn batch(&self, items: Vec<ImageDatasetItem>, device: &B::Device) -> ClassificationBatch<B> {
        fn image_as_vec_u8(item: ImageDatasetItem) -> Vec<u8> {
            // Convert Vec<PixelDepth> to Vec<u8> (we know that CIFAR images are u8)
            item.image
                .into_iter()
                .map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
                .collect::<Vec<u8>>()
        }

        let targets = items.iter().map(|item| {
            if let Annotation::Label(y) = item.annotation {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(y as i64).elem::<B::IntElem>()]),
                    &device,
                )
            } else {
                panic!("Invalid unit type of targets!");
            }
        })
        .collect();

        let image_paths: Vec<String> = items.iter().map(|item| item.image_path.clone()).collect();

        let images = items.
            into_iter()
            .map(|item| TensorData::new(image_as_vec_u8(item), Shape::new([IMG_SIZE, IMG_SIZE, CHANNELS])))
            .map(|data| {
                Tensor::<B, 3>::from_data(data, &device)
                .swap_dims(2, 1)
                .swap_dims(0, 1)
            })
            .map(|tensor| {
                tensor / 255
            })
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);
        
        ClassificationBatch {
            images,
            targets,
            image_paths
        }
    }   
}