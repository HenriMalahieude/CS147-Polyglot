@group(0) @binding(0) var<storage, read> a_matrix: array<u32>;
@group(0) @binding(1) var<storage, read> b_matrix: array<u32>; //NOTE: These must be used by the rest of the program or GRR from "compiler"
@group(0) @binding(2) var<storage, read_write> c_matrix: array<u32>; //		Ok, the reason they need to be used is because otherwise they're basically omitted from the group layout
@group(0) @binding(3) var<storage, read> sizing: array<u32>;//				which causes an error because YAY!!!! Oh and no warning about the reason

@compute
@workgroup_size(16, 16, 1) //This is defining the amount of threads per "thread_block". Note how this api limits to 256
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var row: u32 = global_id.x;
	var col: u32 = global_id.y;

	var m: u32 = sizing[0];
	var k: u32 = sizing[1];
	var n: u32 = sizing[2];

	if (row < m && col < n){
		var i: u32 = 0u;
		loop {
			if (i >= k){
				break;
			}

			c_matrix[row*n + col] = c_matrix[row*n + col] + (a_matrix[row*k + i] * b_matrix[i*n + col]);

			i = i + 1u;
		}
	}
}