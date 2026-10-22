//! Offscreen frames of the bundled Dust2, drawn by the same world passes
//! the window draws, for looking at what the renderer does without a
//! window: `SNAPSHOT_DIR=<dir> cargo test -p engine --lib snapshot --
//! --ignored` writes one PNG per camera there.

use super::{world, View};
use crate::bsp::Map;
use linalg::{vec3, Vec3};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 640;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// A camera at `eye` turned `yaw` degrees (0 looks along +x, 90 along -z)
/// and `pitch` degrees up.
fn camera(eye: Vec3, yaw: f32, pitch: f32) -> View {
	let (y, p) = (yaw.to_radians(), pitch.to_radians());
	let forward = vec3(p.cos() * y.cos(), p.sin(), -p.cos() * y.sin());
	let right = forward.cross(Vec3::Y).normalize();
	let up = right.cross(forward);
	View {
		eye,
		forward,
		right,
		up,
		// 90 degrees across at 4:3, as the game's default `fov`.
		fovy: 2.0 * 0.75f32.atan(),
	}
}

/// The first `classname` entity's origin, engine coordinates.
fn origin_of(path: &std::path::Path, classname: &str) -> Option<Vec3> {
	let bytes = std::fs::read(path).ok()?;
	let entities = crate::bsp::entity_text(&bytes).ok()?;
	let block = entities
		.split('{')
		.find(|b| crate::bsp::field(b, "classname").as_deref() == Some(classname))?;
	let v: Vec<f32> = crate::bsp::field(block, "origin")?
		.split_whitespace()
		.filter_map(|t| t.parse().ok())
		.collect();
	(v.len() == 3).then(|| crate::bsp::to_our([v[0], v[1], v[2]]))
}

fn texture(
	device: &wgpu::Device,
	format: wgpu::TextureFormat,
	extra: wgpu::TextureUsages,
) -> wgpu::Texture {
	device.create_texture(&wgpu::TextureDescriptor {
		label: Some("snapshot"),
		size: wgpu::Extent3d {
			width: WIDTH,
			height: HEIGHT,
			depth_or_array_layers: 1,
		},
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format,
		usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | extra,
		view_formats: &[],
	})
}

/// Draws `view` as the window would with the surfaces pass and `indirect`,
/// and returns its RGBA rows.
fn frame(
	device: &wgpu::Device,
	queue: &wgpu::Queue,
	world: &mut world::World,
	view: &View,
	indirect: world::Indirect,
) -> Vec<u8> {
	let scene =
		texture(device, FORMAT, wgpu::TextureUsages::empty()).create_view(&Default::default());
	let depth = texture(device, DEPTH, wgpu::TextureUsages::empty()).create_view(&Default::default());
	let target = texture(device, FORMAT, wgpu::TextureUsages::COPY_SRC);
	let target_view = target.create_view(&Default::default());
	let size = (WIDTH as f32, HEIGHT as f32);
	let sky = super::SKY;
	let mut encoder = device.create_command_encoder(&Default::default());
	{
		let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
			label: Some("snapshot"),
			color_attachments: &[Some(wgpu::RenderPassColorAttachment {
				view: &scene,
				depth_slice: None,
				resolve_target: None,
				ops: wgpu::Operations {
					load: wgpu::LoadOp::Clear(wgpu::Color {
						r: sky.r as f64,
						g: sky.g as f64,
						b: sky.b as f64,
						a: 1.0,
					}),
					store: wgpu::StoreOp::Store,
				},
			})],
			depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
				view: &depth,
				depth_ops: Some(wgpu::Operations {
					load: wgpu::LoadOp::Clear(0.0),
					store: wgpu::StoreOp::Store,
				}),
				stencil_ops: None,
			}),
			timestamp_writes: None,
			occlusion_query_set: None,
		});
		world.draw(
			queue,
			&mut pass,
			view,
			size,
			sky,
			1.0,
			true,
			0,
			indirect,
			super::world::Fog {
				density: 0.0,
				height: 512.0,
				color: vec3(1.0, 1.0, 1.0),
			},
			true,
		);
	}
	world.occlude(
		device,
		&mut encoder,
		&target_view,
		&scene,
		&depth,
		(WIDTH, HEIGHT),
		indirect.ao > 0.0,
	);
	let row = (WIDTH * 4).next_multiple_of(256);
	let read = device.create_buffer(&wgpu::BufferDescriptor {
		label: Some("snapshot read"),
		size: (row * HEIGHT) as u64,
		usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});
	encoder.copy_texture_to_buffer(
		target.as_image_copy(),
		wgpu::TexelCopyBufferInfo {
			buffer: &read,
			layout: wgpu::TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(row),
				rows_per_image: Some(HEIGHT),
			},
		},
		target.size(),
	);
	queue.submit([encoder.finish()]);
	read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
	device.poll(wgpu::PollType::Wait).unwrap();
	let mapped = read.slice(..).get_mapped_range();
	(0..HEIGHT)
		.flat_map(|y| mapped[(y * row) as usize..(y * row + WIDTH * 4) as usize].to_vec())
		.collect()
}

/// RGBA rows as a PNG: stored deflate blocks, no compression.
fn png(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
	fn crc(bytes: &[u8]) -> u32 {
		let mut c = !0u32;
		for &b in bytes {
			c ^= b as u32;
			for _ in 0..8 {
				c = if c & 1 != 0 {
					0xEDB8_8320 ^ (c >> 1)
				} else {
					c >> 1
				};
			}
		}
		!c
	}
	fn chunk(out: &mut Vec<u8>, kind: &[u8], data: &[u8]) {
		out.extend((data.len() as u32).to_be_bytes());
		let mut body = kind.to_vec();
		body.extend(data);
		out.extend(&body);
		out.extend(crc(&body).to_be_bytes());
	}
	let mut raw = Vec::new();
	for y in 0..height as usize {
		raw.push(0);
		raw.extend(&rgba[y * width as usize * 4..(y + 1) * width as usize * 4]);
	}
	let mut z = vec![0x78, 0x01];
	let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
	for (i, block) in blocks.iter().enumerate() {
		z.push((i + 1 == blocks.len()) as u8);
		z.extend((block.len() as u16).to_le_bytes());
		z.extend((!(block.len() as u16)).to_le_bytes());
		z.extend(*block);
	}
	let (mut a, mut b) = (1u32, 0u32);
	for &byte in &raw {
		a = (a + byte as u32) % 65521;
		b = (b + a) % 65521;
	}
	z.extend(((b << 16) | a).to_be_bytes());
	let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
	let mut header = width.to_be_bytes().to_vec();
	header.extend(height.to_be_bytes());
	header.extend([8, 6, 0, 0, 0]);
	chunk(&mut out, b"IHDR", &header);
	chunk(&mut out, b"IDAT", &z);
	chunk(&mut out, b"IEND", &[]);
	out
}

#[test]
#[ignore]
fn snapshot() {
	let Ok(dir) = std::env::var("SNAPSHOT_DIR") else {
		return;
	};
	let path =
		std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/maps/de_dust2.bsp");
	let map: Map = crate::bsp::load(&path).unwrap();
	let config = crate::lightmap::config(&path).unwrap();
	let (baked, _) = crate::lightmap::bake_or_load(&map, &path).unwrap();
	let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
	let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
	let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
	let mut world = world::World::new(
		&device,
		&queue,
		FORMAT,
		DEPTH,
		&map,
		world::Sources::of(&config),
		&baked,
	);
	let t = origin_of(&path, "info_player_deathmatch").unwrap() + vec3(0.0, 64.0 - 36.0, 0.0);
	let ct = origin_of(&path, "info_player_start").unwrap() + vec3(0.0, 64.0 - 36.0, 0.0);
	let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
	for f in &map.render.faces {
		for v in &f.verts {
			lo = lo.min(v.pos);
			hi = hi.max(v.pos);
		}
	}
	let mid = (lo + hi) * 0.5;
	let cameras = [
		("t_spawn", camera(t, 90.0, -10.0)),
		("t_spawn_back", camera(t, 270.0, -10.0)),
		("ct_spawn", camera(ct, 0.0, -10.0)),
		("ct_spawn_back", camera(ct, 180.0, -10.0)),
		(
			"above",
			camera(vec3(mid.x, hi.y + 600.0, mid.z + 1400.0), 90.0, -50.0),
		),
	];
	let lit = world::Indirect {
		gi: 1.0,
		ao: 1.0,
		radius: 24.0,
		sharp: true,
	};
	let bare = world::Indirect {
		gi: 0.0,
		ao: 0.0,
		radius: 24.0,
		sharp: true,
	};
	// The CPU's own picture of the sun's reach from each camera, ray
	// traced through the same triangles: lit white by the sun's angle,
	// shadowed black, the sky its color. The `_sun` frames must show the
	// same shadows.
	let scene = super::scene::build(&map.render, &[]);
	let bvh = lightbake::Bvh::build(&scene.tris);
	let sun = config.sun_dir.normalize();
	for (name, view) in &cameras {
		let tan = (view.fovy * 0.5).tan();
		let aspect = WIDTH as f32 / HEIGHT as f32;
		let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
		for y in 0..HEIGHT {
			for x in 0..WIDTH {
				let sx = (2.0 * (x as f32 + 0.5) / WIDTH as f32 - 1.0) * tan * aspect;
				let sy = (1.0 - 2.0 * (y as f32 + 0.5) / HEIGHT as f32) * tan;
				let d = (view.forward + view.right * sx + view.up * sy).normalize();
				let px = match bvh.cast(&scene.tris, view.eye, d, 1e30) {
					None => [158, 181, 209],
					Some((i, t)) => {
						// The face's own outward normal, as the surfaces pass
						// lights it: a face seen from behind — the map
						// from outside — faces away from the eye.
						let n = physics::triangle_normal(&scene.tris[i]);
						let p = view.eye + d * t + n * 0.125;
						let n_l = n.dot(sun).max(0.0);
						let lit = n_l > 0.0 && bvh.cast(&scene.tris, p, sun, 1e30).is_none();
						let v = if lit { (60.0 + 195.0 * n_l) as u8 } else { 0 };
						[v, v, v]
					}
				};
				rgba.extend([px[0], px[1], px[2], 255]);
			}
		}
		let file = std::path::Path::new(&dir).join(format!("{name}_cpu.png"));
		std::fs::write(&file, png(&rgba, WIDTH, HEIGHT)).unwrap();
	}
	world.set_dynamic(&queue, &[], &[], true, true);
	for (name, view) in cameras {
		for (kind, indirect) in [("lit", lit), ("sun", bare)] {
			let rgba = frame(&device, &queue, &mut world, &view, indirect);
			let file = std::path::Path::new(&dir).join(format!("{name}_{kind}.png"));
			std::fs::write(&file, png(&rgba, WIDTH, HEIGHT)).unwrap();
		}
	}
	// The moving things: a player-sized capsule and a sphere standing on
	// the floor ahead of the T spawn, its shadow cast over the world's, and
	// a warm light beside them; the same view without them, and with plain
	// diffuse surfaces.
	let view = camera(t, 270.0, -20.0);
	let floor = t.y - 64.0;
	let at = t + vec3(-40.0, 0.0, 140.0);
	let capsule = (
		vec3(at.x, floor + 11.0, at.z),
		vec3(at.x, floor + 61.0, at.z),
		11.0,
	);
	let ball = (vec3(at.x + 50.0, floor + 8.0, at.z - 20.0), 8.0);
	let light = world::DynLight {
		pos: vec3(at.x + 60.0, floor + 40.0, at.z - 60.0),
		color: vec3(6.0, 4.2, 2.0),
		reach: 400.0,
	};
	world.set_bodies(&queue, &[ball]);
	for (kind, pbr, dynamic) in [
		("dynamic", true, true),
		("static", true, false),
		("diffuse", false, true),
	] {
		let (lights, shapes) = if dynamic {
			(vec![light], vec![capsule, (ball.0, ball.0, ball.1)])
		} else {
			(vec![], vec![])
		};
		world.set_dynamic(&queue, &lights, &shapes, pbr, true);
		let rgba = frame(&device, &queue, &mut world, &view, lit);
		let file = std::path::Path::new(&dir).join(format!("moving_{kind}.png"));
		std::fs::write(&file, png(&rgba, WIDTH, HEIGHT)).unwrap();
	}
}
