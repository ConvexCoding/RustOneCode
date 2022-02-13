use std::ops::{Add, Sub, Mul, Div};
use rand::Rng;

#[repr(C)]
pub struct GenRays
{
	baserays: u32,
	num_angles: u32,
	half_ap: f64,
	half_ang: f64
}

#[repr(C)]
pub struct Side
{
	r: f64,
	c: f64,
	k: f64,
	ad: f64,
	ae: f64,
	surf_type: i32 // 0 = plane, 1 = sphere with or wo conic, 2 = poly asphere
}

#[repr(C)]
pub struct Lens
{
	diameter: f64,
	//material: String,
	//lens_type: String,
	ap: f64,
	n: f64,
	wl: f64,
	ct: f64,
	side1: Side,
	side2: Side,
	efl: f64,
	bfl: f64
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vector3D 
{
    x: f64,
    y: f64,
    z: f64
}

#[repr(C)]
pub struct Ray
{
	pvector: Vector3D,
	edir:    Vector3D,
}

impl Vector3D
{
	fn length(self) -> f64
	{
		(self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
	}

	fn _lengthsquared(self) -> f64
	{
		self.x * self.x + self.y * self.y + self.z * self.z
	}
}

impl Add for Vector3D
{
	type Output = Self;
    fn add(self, other: Self) -> Self 
	{
        Self {x: self.x + other.x, y: self.y + other.y, z: self.z + other.z}
	}
}

impl Sub for Vector3D 
{
	type Output = Self;
    fn sub(self, other: Self) -> Self 
	{
        Self {x: self.x - other.x, y: self.y - other.y, z: self.z - other.z}
	}
}

impl Mul<f64> for Vector3D
{
	type Output = Self;
    fn mul(self, other: f64) -> Self 
	{
        Self {x: self.x * other, y: self.y * other, z: self.z * other}
	}
}

impl Mul<Vector3D> for Vector3D
{
	type Output = Self;
    fn mul(self, other: Self) -> Self 
	{
        Self {x: self.x * other.x, y: self.y * other.y, z: self.z * other.z}
	}
}

impl Div<f64> for Vector3D
{
	type Output = Self;
    fn div(self, other: f64) -> Self 
	{
        Self {x: self.x / other, y: self.y / other, z: self.z / other}
	}
}

//
// ===================================================================================
// External Functions visible over FFI
//

#[no_mangle]
pub extern fn tracerays(ptrin: *mut Ray, prtout: *mut Ray, npts: usize, lens: Lens, refocus: f64, aposi: usize) -> f64
{
	let din:  &mut [Ray] = unsafe { std::slice::from_raw_parts_mut(ptrin, npts) }; 
	let dout: &mut [Ray] = unsafe { std::slice::from_raw_parts_mut(prtout, npts) }; 
	
	for i in 0..npts
	{
		let tr = trace_ray(din[i].pvector, din[i].edir, &lens, refocus);	
		dout[i].pvector = tr.pvector;
		dout[i].edir = tr.edir;
	}
	
	return dout[aposi].pvector.x;
}

#[no_mangle]
pub fn gen_random_rays(gr: GenRays, din: &mut [Ray]) 
{
	let mut x: f64;
	let mut y: f64;
	let mut xdir: f64;
	let mut ydir: f64;
	
	let diag = gr.half_ap * gr.half_ap;
	let anglediag = gr.half_ang * gr.half_ang;
	
	let mut count: usize = 0;
	
	let mut rng = rand::thread_rng();
	
	for _i in 0..gr.baserays
	{
		x = rng.gen_range(-gr.half_ap, gr.half_ap);
		y = rng.gen_range(-gr.half_ap, gr.half_ap);
		while (x*x + y*y) > diag
		{
			x = rng.gen_range(-gr.half_ap, gr.half_ap);
			y = rng.gen_range(-gr.half_ap, gr.half_ap);
		}
		let pvbase = Vector3D{x: x, y: y, z: 0.0};
		
		for _j in 0..gr.num_angles
		{
			xdir = rng.gen_range(-gr.half_ang, gr.half_ang);
			ydir = rng.gen_range(-gr.half_ang, gr.half_ang);
			while (xdir*xdir + ydir*ydir) > anglediag
			{
				xdir = rng.gen_range(-gr.half_ang, gr.half_ang);
				ydir = rng.gen_range(-gr.half_ang, gr.half_ang);
			}
			let edir = Vector3D{x: xdir, y: ydir, z: (1.0 - xdir * xdir - ydir * ydir).sqrt()};
			din[count].pvector = pvbase;
			din[count].edir = edir;
			count += 1;
		}
	}
}

#[no_mangle]
pub extern fn gen_trace_rays(gr: GenRays, ptrin: *mut Ray, prtout: *mut Ray, npts: usize, lens: Lens, refocus: f64) -> bool 
{
	let din:  &mut [Ray] = unsafe { std::slice::from_raw_parts_mut(ptrin, npts) }; 
	let dout: &mut [Ray] = unsafe { std::slice::from_raw_parts_mut(prtout, npts) }; 
	
	gen_random_rays(gr, din);
	
	for i in 0..npts
	{
		let tr = trace_ray(din[i].pvector, din[i].edir, &lens, refocus);	
		dout[i].pvector = tr.pvector;
		dout[i].edir = tr.edir;
	}
	
	return true;
}

#[no_mangle]
pub extern fn trace_ray(p0: Vector3D, e0: Vector3D, lens: &Lens, refocus: f64) -> Ray
{
	// Trace ray from srf 0 to first lens surface. The axial distance here should be zero.
	let p2 = translate_to_surface(p0, e0, &lens.side1, 0.0);
	let n2 = calc_slope(p2, &lens.side1);
	let e2 = calc_dir_sines(e0, n2, 1.0, lens.n);  // after refraction
	
	// Trace to Surface 2 after refraction
	let p3 = translate_to_surface(p2, e2, &lens.side2, lens.ct);
	let n3 = calc_slope(Vector3D{x: p3.x, y: p3.y, z: p3.z - lens.ct}, &lens.side2);  // adjust z for CT of lens
	let e3 = calc_dir_sines(e2, n3, lens.n, 1.0);
	
	// transfer ray to image plane
	let p4 = translate_to_flat(p3, e3, lens.ct + lens.bfl + refocus);
	let ofinal = Ray{pvector: p4, edir: e3};
	return ofinal;
}

fn translate_to_surface(p0: Vector3D, e0: Vector3D, side: &Side, plane: f64) -> Vector3D
{
	if side.surf_type == 0
	{
		let pprime = translate_to_flat(p0, e0, plane);
		return pprime;
	}

	let mut zest1 = calc_sag(p0.x, p0.y, &side, 0.001);
	let mut u = (zest1 - p0.z) / e0.z;
	let mut p1 = p0.clone();
	let mut p2 = p0 + e0 * u;

	for _i in 0..10
	{
		if (p1 - p2).length() > 1e-4f64
		{
			p1 = p2;
			zest1 = calc_sag(p1.x, p1.y, &side, 0.001) + plane;
			u = (zest1 - p0.z) / e0.z;
			p2 = p0 + e0 * u;
		}
		else
		{
			break;
		}
	}

	return p2;
}

fn translate_to_flat(p: Vector3D, e: Vector3D, zplane: f64) -> Vector3D
{
	let u = (zplane - p.z) / e.z;
	let pprime = p + e * u;
	return pprime;
}

fn dot_product(v1: Vector3D, v2: Vector3D) -> f64
{
	let x = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	return x;
}

fn calc_slope(p: Vector3D, s: &Side) -> Vector3D 
{
	let r = p.x * p.x + p.y * p.y;
	let q0 = p.z - s.ad * r * r - s.ae * r * r * r;
	let q1 = -4.0 * s.ad * r - 6.0 * s.ae * r * r;

	let dx = p.x * (-s.c - s.c * (s.k + 1.0) * q1 * q0 + q1);
	let dy = p.y * (-s.c - s.c * (s.k + 1.0) * q1 * q0 + q1);
	let dz = 1.0 - s.c * (s.k + 1.0) * q0;

	let mut n = Vector3D{x: dx, y: dy, z: dz};
	n = n / n.length();
	//let f = -(s.c / 2.0) * r - (s.c / 2.0) * (s.k + 1.0) * q0 * q0 + q0;
	return n;
}

fn calc_sag(x: f64, y: f64, side: &Side, rtolforzero: f64) -> f64
{
	
	let mut c = 0.0;
	if side.r.abs() > rtolforzero
	{
		c = 1.0 / side.r;
	}

	let r2 = x * x + y * y;
	let sqrtvalue = 1.0 - (1.0 + side.k) * c * c * r2;

	if sqrtvalue < 0.0
	{
		return 0.0;
	}
	else
	{
		return c * r2 / (1.0 + sqrtvalue.sqrt()) + side.ad * r2 * r2 + side.ae * r2 * r2 * r2;
	}
}

fn calc_dir_sines(ein: Vector3D, ndir: Vector3D, nin: f64, nout: f64) -> Vector3D
{
	let alpha = dot_product(ein, ndir);
	//var aoi = Math.Acos(alpha).RadToDeg();
	//var aor = Math.Asin(Math.Sin(Math.Acos(alpha)) * nin / nout).RadToDeg();

	let a = 1.0;
	let b = 2.0 * alpha;
	let c = 1.0 - (nout * nout) / (nin * nin);
	let sol2 = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
	let mut ep = ein + ndir * sol2;
	ep = ep / ep.length();
	return ep;
}


