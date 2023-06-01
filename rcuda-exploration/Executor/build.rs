use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../Kernel_Code")
        .copy_to("../../resources/add.ptx")
        .build()
        .unwrap();
}