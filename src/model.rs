use burn::{module::Module, nn::{conv::{Conv2d, Conv2dConfig}, pool::{MaxPool2d, MaxPool2dConfig}, BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu}, prelude::Backend, tensor::{activation::softmax, Tensor}};

#[derive(Module, Debug)]
pub struct Lumina<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
    pub pool: MaxPool2d,
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    pub bn1: BatchNorm<B, 2>,
    pub bn2: BatchNorm<B, 2>,
    pub bn3: BatchNorm<B, 2>,
    pub dropout: Dropout,
    pub activation: Relu,
}

impl<B: Backend> Lumina<B> {
    pub fn new(device: &B::Device) -> Lumina<B> {

        // Height x Width x Channels

        Self {
            conv1: Conv2dConfig::new([3, 16], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Valid)
                .init(device),
            bn1: BatchNormConfig::new(16).init(device),
            conv2: Conv2dConfig::new([16, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Valid)
                .init(device),
            bn2: BatchNormConfig::new(32).init(device),
            conv3: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Valid)
                .init(device),
            bn3: BatchNormConfig::new(64).init(device),
            pool: MaxPool2dConfig::new([2,2])
                .with_strides([2, 2])
                .init(),
            fc1: LinearConfig::new(13*13*64, 128)
                .init(device),  
            dropout: DropoutConfig::new(0.5).init(),
            fc2: LinearConfig::new(128, 7)
                .init(device),  
            activation: Relu::new(),
        }
    }

    // img shape - [num of images, rgb channels, cols, rows]
    // output shape - [num o images, num of classes]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _, _, _] = images.dims();

        //Conv1
        let x = self.conv1.forward(images);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        //Conv2
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x);
        
        //Conv3
        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.activation.forward(x);
        
        //Double pool for parameter reduction
        let x = self.pool.forward(x);
        let x = self.pool.forward(x);

        //Reshape 
        let x = x.reshape([batch_size, 13*13*64]);

        //FC
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        // let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }

    pub fn print(&self) {
        println!("{}", self);
    }
}