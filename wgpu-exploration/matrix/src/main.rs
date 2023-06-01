use std::{time::Instant, borrow::Cow};
use wgpu::util::DeviceExt;
use rand::Rng;


#[macro_export]
macro_rules! rm_access {
	($arr:ident, $row:ident, $col:ident, $col_dim:ident) => {
		$arr[($row as usize) * ($col_dim as usize) + ($col as usize)]
	};
}

const M: u32 = 500; //Ran successfully with 1000 x 1000 x 1000, but took forever
const K: u32 = 500;
const N: u32 = 500;

fn output_matrix(mat: &Vec<u32>, m: usize, n: usize) {
	println!("Matrix ({}) {m:?} x {n:?}", mat.len());
	for row in 0..m {
		let mut mess: String = String::from("");
		for col in 0..n {
			mess.push_str(&mat[row*n + col].to_string());
			mess.push(' ');
		}
		println!("{}", mess);
	}
	println!("\n");
}

fn generate_matrix(x: u32, y: u32) -> Vec<u32> {
	let mut mat: Vec<u32> = vec![];
	for _i in 1..=(x*y) {
		let rr = rand::thread_rng().gen_range(1..=5) as u32;
		mat.push(rr);
	}

	mat //A bit of syntactical sugar makes the return go down
}

fn verify(a: &Vec<u32>, b: &Vec<u32>, c: &Vec<u32>) -> bool {
	for col in 0..N {
		for row in 0..M {
			//Now for each element, we calculate the resultant
			let mut cpu_val: u32 = 0;
			for j in 0..K {
				cpu_val += rm_access!(a, row, j, K) * rm_access!(b, j, col, N);
			}

			let gpu_val = rm_access!(c, row, col, N);

			if cpu_val != gpu_val {
				println!("Error at ({row:?}, {col:?}) of C: {cpu_val:?} vs {gpu_val:?}");
				return false;
			}
		}
	}

	return true;
}

async fn run(){
	let matrix_a: Vec<u32> = generate_matrix(M, K); //m x k, m rows, and k columns
	let matrix_b: Vec<u32> = generate_matrix(K, N);

	//output_matrix(&matrix_a, M as usize, K as usize);
	//output_matrix(&matrix_b, K as usize, N as usize);
	
	let compute_then = Instant::now();

	//Now we run the gpu code
	let matrix_c: Option<Vec<u32>> = gpu_prep(&matrix_a, &matrix_b).await;

	let compute_now = compute_then.elapsed().as_millis();
	println!("Compute Shader took: {compute_now:?}ms");

	match matrix_c {
		Some(c) => {
			let verify_then = Instant::now();

			//output_matrix(&c, M as usize, N as usize);

			let status: bool = verify(&matrix_a, &matrix_b, &c);

			let verify_now = verify_then.elapsed().as_millis();
			println!("Verification Status: {status:?}\n\t\tTook: {verify_now:?}ms");
		},
		None => println!("GPU Failed to compute?"),
	};
}

async fn gpu_prep(a: &Vec<u32>, b: &Vec<u32>) -> Option<Vec<u32>> {
	let wgpu_instance = wgpu::Instance::default();

	let device_adapter = wgpu_instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;

	let (device, queue) = device_adapter
		.request_device(
			&wgpu::DeviceDescriptor{
				label: None, 
				features: wgpu::Features::empty(), 
				limits: wgpu::Limits::downlevel_defaults()
			},
			None
		)
		.await
		.unwrap();
	
	//No Idea, but We'll copy this in too maybe
	let info = device_adapter.get_info();
	// skip this on LavaPipe temporarily
	if info.vendor == 0x10005 {
		return None;
	}

	gpu_kernel_launcher(&device, &queue, &a, &b).await
}

async fn gpu_kernel_launcher(device: &wgpu::Device, queue: &wgpu::Queue, a: &Vec<u32>, b: &Vec<u32>) -> Option<Vec<u32>>{
	// Loads the shader from WGSL
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

	let size_c = ((M * N) as u64) * std::mem::size_of::<u32>() as wgpu::BufferAddress;

	//Location we'll be copying data to on the CPU back from the GPU
	let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_c,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

	//Set up the inputs to the shader
	let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { //Copy of what we'll be sending
        label: Some("Storage A"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

	let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage B"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

	//You may be wondering why we have an extra buffer here, honestly, idk either
	//Ok but seriously it's used to be a "location" to extract the resulting matrix from. I'm really starting to dislike wgpu
	//I don't want to test it, but I wonder if I can just put the staging_buffer in the binding layout below instead of this
	let buffer_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Storage C"),
		contents: bytemuck::cast_slice(&vec![0 as u32; (M*N) as usize]),
		usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
	});

	let buffer_sizes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Sizes"),
        contents: bytemuck::cast_slice(&vec![M, K, N]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

	// Instantiates the pipeline, I'm getting really tired of the boilerplate. . . .
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: "main",
    });

	// Instantiates the bind group, to bind shader variables into our own code
    let bind_group_layout0 = compute_pipeline.get_bind_group_layout(0);
    let bind_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Group 0"),
        layout: &bind_group_layout0,
        entries: &[
			wgpu::BindGroupEntry {
				binding: 0,
				resource: buffer_a.as_entire_binding(),
			},
			
			wgpu::BindGroupEntry {
				binding: 1,
				resource: buffer_b.as_entire_binding(),
			},

			wgpu::BindGroupEntry {
				binding: 2,
				resource: buffer_c.as_entire_binding(),
			},
			
			wgpu::BindGroupEntry {
				binding: 3,
				resource: buffer_sizes.as_entire_binding(),
			}
		],
    });

	//Almost copied word for word from example
	let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group0, &[]); //Now that we've created the group, we have to set it....
		//cpass.set_bind_group(1, &bind_group1, &[]);
        cpass.insert_debug_marker("matrix multiplication");

		//analagous to: kernel<<<dim3{(M/16) + 1, (N/16) + 1, 1}, dim3{?, ?, ?}>>>
		//So here it does the thread blocks
		let block_dim_x = (M/16) + 1;
		let block_dim_y = (N/16) + 1;
		println!("Launching {block_dim_x:?} x {block_dim_y:?} thread blocks of 16 x 16");
        cpass.dispatch_workgroups(block_dim_x, block_dim_y, 1);
    }
	encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging_buffer, 0, size_c);

	//Did you think we were done with boilerplate?
	queue.submit(Some(encoder.finish())); //Now we gotta actually run our request x_x

	// Note that we're not calling `.await` here.
	let buffer_slice = staging_buffer.slice(..);
	// Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
	let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
	buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

	//This can be compared to __deviceSynchronize() in cuda
	device.poll(wgpu::Maintain::Wait);

	//I've copied the code from the example again, because I hate typing
	// Awaits until `buffer_future` can be read from
    if let Some(Ok(())) = receiver.receive().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        return Some(result);
    } else {
        panic!("\tfailed to run compute on gpu!")
    }
}

fn main() {
	let now = Instant::now();

	pollster::block_on(run());

	let then = now.elapsed().as_millis();
	println!("Total Time to run: {then:?}ms");
}