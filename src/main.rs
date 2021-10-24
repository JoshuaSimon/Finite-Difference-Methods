use std::fs::File;
use std::io::prelude::*;
use std::io::LineWriter;
use std::time::{Duration, Instant};
use ndarray::{Array, Array1, Array2, ArrayView1, Axis, s};
//use ndarray::prelude::*;
use ndarray_linalg::Solve;
//use nalgebra;

struct Diffusion {
    alpha: f32,                     // thermal diffusivity, positive constant
    delta_x: f32,                   // Step length for spatial discretization
    delta_t: f32,                   // Step length for time discretization
    n_x: usize,                     // Number of spatial points on the grid
    n_t: usize,                     // Number of time steps on the grid
    r: f32,                         // Constant for easier use when setting up system matrix
    a_matrix: Array2<f32>,          // A-Matrix form A * z = b
    b_vector: Array1<f32>,          // b-Vector from A * z = b
    z_matrix: Array2<f32>,          // Matrix contaning all solution vectors z from A * z = b
}

impl Diffusion {
    /// Constructor
    fn new(alpha: f32, delta_x: f32, delta_t: f32, n_x: usize, n_t: usize) -> Diffusion {
        Diffusion {
            alpha: alpha,
            delta_x: delta_x,
            delta_t: delta_t,
            n_x: n_x,
            n_t: n_t,
            r: alpha * delta_t  / (delta_x.powi(2) * 2.0),
            a_matrix: Array::eye(n_x),
            b_vector: Array::zeros(n_x),
            z_matrix: Array::zeros((0, n_x)),
        }
    }

    /// Assemble the b-Vector and set bounary conditions
    /// Takes the temperature vector form the previous iteration
    /// as input.
    fn assemble_b_vector(&mut self, temp: Array1<f32>) {
        for i in 1..self.n_x - 1 {
            self.b_vector[[i]] = self.r * temp[i - 1] + (1.0 - 2.0 * self.r) * temp[i] + self.r * temp [i + 1];
        }

        // Is this correct?
        //self.b_vector[[0]] = self.b_vector[[0]] + self.r * temp[0];
        //self.b_vector[[self.n_x - 1]] = self.b_vector[[self.n_x - 1]] + self.r * temp[self.n_x - 1];

        self.b_vector[[0]] = temp[0];
        self.b_vector[[self.n_x - 1]] = temp[self.n_x - 1];
    }

    /// Construct system matrix add boundary conditions.
    fn add_boundaries(&mut self, t_ambience: f32, t_0: f32, t_n: f32) {
        // Assemble temperature vector from start values.
        //let mut temp: Vec<f32> = vec![t_ambience; self.n_x]; 
        let mut temp: Array1<f32> = Array::ones(self.n_x) * t_ambience;
        temp[[0]] = t_0;
        temp[[self.n_x - 1]] = t_n;

        // Assemble the A-Matrix to set bounary conditions
        //let d: f32 = -2.0 * (1.0 + 1.0 / self.r); // Wimmer's shortcut
        //self.a_matrix = &self.a_matrix * d;
        self.a_matrix = &self.a_matrix * (1.0 + 2.0 * self.r);

        // Iterrate over rows and columns of the A-Matrix
        // to set the side-diagonal elements to 1.0.
        // First and last row are skipped, because they hold
        // the boundary conditions.
        for row in 1..self.n_x - 1 {
            for col in 0..self.n_x {
                    if col == row - 1 {
                        self.a_matrix[[row, col]] = -1.0 * self.r;
                    } else if col == row + 1 {
                        self.a_matrix[[row, col]] = -1.0 * self.r;
                }
            }
        }

        // Set boundary conditions to A-Matrix
        self.a_matrix[[0, 0]] = 1.0;
        self.a_matrix[[self.n_x - 1, self.n_x - 1]] = 1.0;

        // Assemble the b-Vector and set bounary conditions
        //self.assemble_b_vector(temp);
        self.b_vector = temp;
    }

    /// Solve the matrix equation A * z = b for the time discretization.
    fn solve(&mut self) {
        //println!("Temprature on t = {}: {:?}", "start", self.b_vector);
        let mut _shape_erorr = self.z_matrix.push_row(self.b_vector.view());

        // Iteratre through time discretization and calculate the
        // new temprature vector within every iteration.
        for t in 0..self.n_t {
            let b = self.b_vector.clone();
            let z = self.a_matrix.solve_into(b).unwrap();
            self.assemble_b_vector(z);

            // Store the solution for the current iteration.
            //self.z_matrix.row_mut(t)[t] = z.view_mut();
            _shape_erorr = self.z_matrix.push_row(self.b_vector.view());

            //if t % 10 == 0 {
            //    println!("Temprature on t = {}: {:?}", t, self.b_vector);
            //}
        }
    }

    /// Write the solution to an output file.
    fn wirte_output_file(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let mut file = LineWriter::new(file);
        let mut line_str: String;
        
        for row in self.z_matrix.rows() {
            for entry in row {
                line_str = format!("{:.6}", entry).to_owned();
                line_str.push_str(", ");
                file.write_all(line_str.as_bytes())?;
            }
            file.write_all(b"\n")?;
        } 

        file.flush()?;
        Ok(())
    }
}


fn main() {
    let start = Instant::now();
    let mut model = Diffusion::new(1.0, 0.5, 0.5, 100, 2000);
    model.add_boundaries(20.0, 50.0, 80.0);
    model.solve();
    let duration = start.elapsed();

    let _output_erorr = model.wirte_output_file("temperature_output.txt");

    //println!("A-Matrix: {:?}", model.a_matrix);
    //println!("b-Vector: {:?}", model.b_vector);
    println!("Z-Matrix: {:?}", model.z_matrix);

    println!("Finished in {:?}.", duration);
}
