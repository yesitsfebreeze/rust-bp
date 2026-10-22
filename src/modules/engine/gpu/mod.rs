//! The game's window, input and renderer, on winit and wgpu. The map draws
//! as its own surfaces, triangle by triangle (see `raster.wgsl`), or as
//! whole voxels: voxelized ahead of the view into sparse shells around
//! the eye and marched per pixel (see `shell.wgsl`), or traced per hit
//! instead (see `trace.wgsl`). The world draws into the frame's own color
//! and depth, which a full-screen pass then shades darker where the depth
//! shows matter standing near (see `ao.wgsl`) and carries to the window.
//! Over
//! that go world-space lines and boxes, the textured viewmodel, and the
//! flat HUD with its text.
//!
//! [`Window::next_frame`] drives the frame loop the way an immediate-mode
//! library does: it presents what [`Gfx`] recorded, pumps the window's
//! events into [`Input`], and returns false once the window closes.

mod ao;
mod exposure;
mod overlay;
mod raster;
mod scene;
mod shell;
#[cfg(test)]
mod snapshot;
mod text;
mod timing;
mod util;
mod world;

use linalg::{vec3, Vec3};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{CursorGrabMode, WindowId};

pub use overlay::{TextureId, Vertex};
pub use winit::event::MouseButton;
pub use winit::keyboard::KeyCode as Key;
pub use world::{WorldStats, MAX_DENTS};

/// What the trace can show, in the order [`Gfx::cycle_view_mode`] steps
/// through: the lit world, then debug views of the cubes it found — depth
/// (white near, black far), the BSP plane's normal, albedo, one color per
/// cube, one color per cube size.
pub const VIEW_MODES: [&str; 6] = ["lit", "depth", "normal", "albedo", "cubes", "cube size"];

/// The sky the map is drawn against, and lit by in the bake.
pub const SKY: Color = Color::new(0.62, 0.71, 0.82, 1.0);

/// A straight-alpha color, its channels as the screen shows them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
	pub r: f32,
	pub g: f32,
	pub b: f32,
	pub a: f32,
}

impl Color {
	pub const WHITE: Color = Color::new(1.0, 1.0, 1.0, 1.0);
	pub const BLACK: Color = Color::new(0.0, 0.0, 0.0, 1.0);
	pub const GREEN: Color = Color::new(0.0, 0.89, 0.19, 1.0);

	pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
		Color { r, g, b, a }
	}
}

/// This frame's input: what is held, and what happened since the last frame.
#[derive(Clone, Default)]
pub struct Input {
	down: HashSet<Key>,
	pressed: HashSet<Key>,
	buttons: HashSet<MouseButton>,
	clicked: HashSet<MouseButton>,
	chars: Vec<char>,
	mouse: (f32, f32),
	cursor: (f32, f32),
	wheel: f32,
	focused: bool,
}

impl Input {
	pub fn down(&self, key: Key) -> bool {
		self.down.contains(&key)
	}

	/// Pressed since the last frame; key repeat does not count.
	pub fn pressed(&self, key: Key) -> bool {
		self.pressed.contains(&key)
	}

	pub fn button_down(&self, button: MouseButton) -> bool {
		self.buttons.contains(&button)
	}

	pub fn clicked(&self, button: MouseButton) -> bool {
		self.clicked.contains(&button)
	}

	/// Text typed since the last frame.
	pub fn chars(&self) -> &[char] {
		&self.chars
	}

	/// Raw mouse counts since the last frame, unaffected by the cursor grab.
	pub fn mouse_delta(&self) -> (f32, f32) {
		self.mouse
	}

	/// Where the cursor is, pixels from the window's top left.
	pub fn cursor(&self) -> (f32, f32) {
		self.cursor
	}

	/// Wheel notches since the last frame.
	pub fn wheel(&self) -> f32 {
		self.wheel
	}

	/// Whether the window has the keyboard focus.
	pub fn focused(&self) -> bool {
		self.focused
	}

	/// Forgets what happened, keeping what is held.
	fn begin(&mut self) {
		self.pressed.clear();
		self.clicked.clear();
		self.chars.clear();
		self.mouse = (0.0, 0.0);
		self.wheel = 0.0;
	}
}

/// The window, its input, and its renderer.
pub struct Window {
	events: EventLoop<()>,
	shell: Shell,
	last: Instant,
	frame_time: f32,
	fps: u32,
	frames: u32,
	second: Instant,
}

struct Shell {
	title: String,
	size: (u32, u32),
	window: Option<Arc<winit::window::Window>>,
	gfx: Option<Gfx>,
	input: Input,
	error: Option<String>,
	closed: bool,
	grabbed: bool,
}

impl Window {
	/// Opens a window `width` by `height` pixels with its GPU renderer.
	pub fn open(title: &str, width: u32, height: u32) -> Result<Window, String> {
		let events = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
		let mut window = Window {
			events,
			shell: Shell {
				title: title.to_string(),
				size: (width, height),
				window: None,
				gfx: None,
				input: Input {
					focused: true,
					..Default::default()
				},
				error: None,
				closed: false,
				grabbed: false,
			},
			last: Instant::now(),
			frame_time: 1.0 / 60.0,
			fps: 0,
			frames: 0,
			second: Instant::now(),
		};
		// The window comes with the loop's first resume.
		while window.shell.gfx.is_none() {
			window.pump();
			if let Some(e) = window.shell.error.take() {
				return Err(e);
			}
			if window.shell.closed {
				return Err("the window closed while opening".into());
			}
		}
		Ok(window)
	}

	fn pump(&mut self) {
		let _ = self
			.events
			.pump_app_events(Some(Duration::ZERO), &mut self.shell);
	}

	/// Presents the frame [`Gfx`] recorded, then gathers the next frame's
	/// input. False once the window is closed.
	pub fn next_frame(&mut self) -> bool {
		if let Some(gfx) = &mut self.shell.gfx {
			gfx.present();
		}
		self.shell.input.begin();
		self.pump();
		let now = Instant::now();
		self.frame_time = (now - self.last).as_secs_f32();
		self.last = now;
		self.frames += 1;
		if self.second.elapsed() >= Duration::from_secs(1) {
			self.fps = self.frames;
			self.frames = 0;
			self.second = now;
		}
		!self.shell.closed
	}

	pub fn input(&self) -> &Input {
		&self.shell.input
	}

	pub fn gfx(&mut self) -> &mut Gfx {
		self.shell.gfx.as_mut().expect("opened with a renderer")
	}

	/// Seconds the last frame took.
	pub fn frame_time(&self) -> f32 {
		self.frame_time
	}

	/// Frames presented in the last whole second.
	pub fn fps(&self) -> u32 {
		self.fps
	}

	/// Locks and hides the cursor for mouse look, or frees it.
	pub fn grab_cursor(&mut self, grab: bool) {
		if self.shell.grabbed == grab {
			return;
		}
		self.shell.grabbed = grab;
		if let Some(window) = &self.shell.window {
			let _ = if grab {
				window
					.set_cursor_grab(CursorGrabMode::Locked)
					.or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
			} else {
				window.set_cursor_grab(CursorGrabMode::None)
			};
			window.set_cursor_visible(!grab);
		}
	}
}

impl ApplicationHandler for Shell {
	fn resumed(&mut self, events: &ActiveEventLoop) {
		if self.window.is_some() {
			return;
		}
		let attributes = winit::window::Window::default_attributes()
			.with_title(&self.title)
			.with_inner_size(winit::dpi::PhysicalSize::new(self.size.0, self.size.1));
		let window = match events.create_window(attributes) {
			Ok(window) => Arc::new(window),
			Err(e) => {
				self.error = Some(format!("window: {e}"));
				return;
			}
		};
		match Gfx::new(window.clone()) {
			Ok(gfx) => self.gfx = Some(gfx),
			Err(e) => self.error = Some(e),
		}
		self.window = Some(window);
	}

	fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
		let input = &mut self.input;
		match event {
			WindowEvent::CloseRequested => self.closed = true,
			WindowEvent::Resized(size) => {
				if let Some(gfx) = &mut self.gfx {
					gfx.resize(size.width, size.height);
				}
			}
			WindowEvent::Focused(focused) => {
				input.focused = focused;
				if !focused {
					input.down.clear();
					input.buttons.clear();
				}
			}
			WindowEvent::KeyboardInput { event, .. } => {
				if event.state == ElementState::Pressed {
					if let Some(text) = &event.text {
						input.chars.extend(text.chars());
					}
				}
				let PhysicalKey::Code(key) = event.physical_key else {
					return;
				};
				match event.state {
					ElementState::Pressed => {
						if !event.repeat {
							input.pressed.insert(key);
						}
						input.down.insert(key);
					}
					ElementState::Released => {
						input.down.remove(&key);
					}
				}
			}
			WindowEvent::MouseInput { state, button, .. } => match state {
				ElementState::Pressed => {
					input.buttons.insert(button);
					input.clicked.insert(button);
				}
				ElementState::Released => {
					input.buttons.remove(&button);
				}
			},
			WindowEvent::CursorMoved { position, .. } => {
				input.cursor = (position.x as f32, position.y as f32);
			}
			WindowEvent::MouseWheel { delta, .. } => {
				input.wheel += match delta {
					MouseScrollDelta::LineDelta(_, y) => y,
					MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
				};
			}
			_ => {}
		}
	}

	fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
		if let DeviceEvent::MouseMotion { delta } = event {
			self.input.mouse.0 += delta.0 as f32;
			self.input.mouse.1 += delta.1 as f32;
		}
	}
}

/// Where the world is seen from this frame.
#[derive(Clone, Copy, Debug)]
pub struct View {
	pub eye: Vec3,
	pub forward: Vec3,
	pub right: Vec3,
	pub up: Vec3,
	/// Vertical field of view, radians.
	pub fovy: f32,
}

/// The renderer: the traced world and the frame's overlays.
pub struct Gfx {
	window: Arc<winit::window::Window>,
	surface: wgpu::Surface<'static>,
	device: wgpu::Device,
	queue: wgpu::Queue,
	config: wgpu::SurfaceConfiguration,
	/// Whether shaders must sRGB-encode: the swapchain does not.
	encode: bool,
	depth: wgpu::TextureView,
	/// The frame's own color, drawn by the world pass and read back for its
	/// ambient occlusion.
	scene: wgpu::TextureView,
	world: Option<world::World>,
	overlay: overlay::Overlay,
	timer: Option<timing::Timer>,
	view: Option<View>,
	/// The sky's color, as the screen shows it, and the scene's exposure.
	sky: Color,
	exposure: f32,
	/// Pixels a voxel spans at any distance.
	voxel: f32,
	/// Which of [`VIEW_MODES`] the trace shows.
	view_mode: usize,
	/// Rays the trace averages per pixel: 1, 2 or 4.
	rays: u32,
	/// Whether the frame's edges are smoothed as it is carried to the
	/// window (FXAA).
	smooth: bool,
	/// Whether the map draws as its own surfaces, the triangles it is made
	/// of, rasterized at the screen's native resolution.
	surfaces: bool,
	/// Whether the map is marched through the shell field instead of traced.
	shell: bool,
	/// Whether the shell field's cubes draw their edges pink.
	wire: bool,
	/// Where the shell field's cube sizes change over distance.
	lod: world::Lod,
	/// How the baked ambient light and its occlusion land.
	indirect: world::Indirect,
	/// The air the surface view is fogged by.
	fog: world::Fog,
	/// The eye's adaptation, and whether it adapts.
	meter: exposure::Meter,
	auto_exposure: bool,
	/// The range every frame's exposure stays in.
	exposure_range: (f32, f32),
	/// This frame's dynamic lights and moving shapes, recorded as the
	/// overlays are and cleared after the frame.
	lights: Vec<world::DynLight>,
	occluders: Vec<(Vec3, Vec3, f32)>,
	/// This frame's loose faces, packed as surface corners.
	loose: Vec<u8>,
	/// Whether surfaces reflect by their finish, whether dynamic lights
	/// light, and whether moving shapes cast shadows.
	pbr: bool,
	dynamic_lights: bool,
	dynamic_shadows: bool,
}

/// Ends the stage named `label`, when the frame is timed.
fn mark(
	timer: &mut Option<timing::Timer>,
	encoder: &mut wgpu::CommandEncoder,
	label: &'static str,
) {
	if let Some(timer) = timer {
		timer.mark(encoder, label);
	}
}

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// The near plane both the trace and the overlays project with.
pub const NEAR: f32 = 1.0;

fn depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
	device
		.create_texture(&wgpu::TextureDescriptor {
			label: Some("depth"),
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: DEPTH,
			// Also bound as a texture by the ambient occlusion pass.
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		})
		.create_view(&Default::default())
}

/// The frame's own color, the world pass draws into so its pixels can be
/// read back and shaded before the window.
fn scene_view(
	device: &wgpu::Device,
	format: wgpu::TextureFormat,
	width: u32,
	height: u32,
) -> wgpu::TextureView {
	device
		.create_texture(&wgpu::TextureDescriptor {
			label: Some("scene"),
			size: wgpu::Extent3d {
				width,
				height,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
			view_formats: &[],
		})
		.create_view(&Default::default())
}

impl Gfx {
	fn new(window: Arc<winit::window::Window>) -> Result<Gfx, String> {
		let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
		let surface = instance
			.create_surface(window.clone())
			.map_err(|e| format!("surface: {e}"))?;
		let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::HighPerformance,
			compatible_surface: Some(&surface),
			force_fallback_adapter: false,
		}))
		.map_err(|e| format!("no GPU adapter: {e}"))?;
		let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
			label: Some("gfx"),
			// Stage timings when the adapter can write timestamps between
			// passes.
			required_features: adapter.features() & timing::Timer::FEATURES,
			// The shell's face buffers bind whole: half the shells' faces
			// stand past the default a storage binding may be.
			required_limits: wgpu::Limits {
				max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
				..Default::default()
			},
			..Default::default()
		}))
		.map_err(|e| format!("no GPU device: {e}"))?;
		let caps = surface.get_capabilities(&adapter);
		let format = caps
			.formats
			.iter()
			.copied()
			.find(|f| !f.is_srgb())
			.or_else(|| caps.formats.first().copied())
			.ok_or("the surface supports no format")?;
		let size = window.inner_size();
		let config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format,
			width: size.width.max(1),
			height: size.height.max(1),
			present_mode: wgpu::PresentMode::AutoNoVsync,
			desired_maximum_frame_latency: 2,
			alpha_mode: caps.alpha_modes[0],
			view_formats: vec![],
		};
		surface.configure(&device, &config);
		let depth = depth_view(&device, config.width, config.height);
		let scene = scene_view(&device, format, config.width, config.height);
		let overlay = overlay::Overlay::new(&device, &queue, format, DEPTH)?;
		let timer = timing::Timer::new(&device, &queue);
		Ok(Gfx {
			window,
			surface,
			encode: !format.is_srgb(),
			depth,
			scene,
			world: None,
			overlay,
			timer,
			view: None,
			sky: SKY,
			exposure: 1.0,
			voxel: 2.0,
			view_mode: 0,
			rays: 1,
			smooth: true,
			surfaces: true,
			shell: true,
			wire: false,
			lod: world::Lod {
				near: 0.0,
				curve: 1.0,
			},
			indirect: world::Indirect {
				gi: 1.0,
				ao: 1.0,
				radius: 24.0,
				traced: true,
			},
			fog: world::Fog {
				density: 0.0,
				height: 512.0,
				color: vec3(1.0, 1.0, 1.0),
			},
			meter: exposure::Meter::new(&device, !format.is_srgb()),
			auto_exposure: true,
			exposure_range: (0.25, 4.0),
			lights: Vec::new(),
			occluders: Vec::new(),
			loose: Vec::new(),
			pbr: true,
			dynamic_lights: true,
			dynamic_shadows: true,
			device,
			queue,
			config,
		})
	}

	fn resize(&mut self, width: u32, height: u32) {
		if width == 0 || height == 0 {
			return;
		}
		self.config.width = width;
		self.config.height = height;
		self.surface.configure(&self.device, &self.config);
		self.depth = depth_view(&self.device, width, height);
		self.scene = scene_view(&self.device, self.config.format, width, height);
	}

	/// GPU milliseconds per stage of recent frames, when the adapter can
	/// time them.
	pub fn timings(&self) -> &[(&'static str, f32)] {
		self.timer.as_ref().map_or(&[], |t| t.results())
	}

	/// The window's size in pixels.
	pub fn size(&self) -> (f32, f32) {
		(self.config.width as f32, self.config.height as f32)
	}

	/// Draws `map` from now on, lit by the bake `baked` holds: its own
	/// direct sun, sky and bounces, sampled as lightmaps by the surface
	/// pass, and by the trace's sun and sky `light` describes.
	pub fn load_world(
		&mut self,
		map: &crate::bsp::Map,
		light: &lightbake::Config,
		baked: &lightbake::Baked,
	) -> WorldStats {
		let world = world::World::new(
			&self.device,
			&self.queue,
			self.config.format,
			DEPTH,
			map,
			world::Sources::of(light),
			baked,
		);
		let stats = world.stats();
		self.world = Some(world);
		stats
	}

	/// Where this frame's world is seen from; without one only the sky and
	/// the HUD draw.
	pub fn set_view(&mut self, view: View) {
		self.view = Some(view);
	}

	pub fn set_sky(&mut self, sky: Color) {
		self.sky = sky;
	}

	pub fn set_exposure(&mut self, exposure: f32) {
		self.exposure = exposure;
	}

	/// Pixels a voxel of the map spans, near or far: the detail setting.
	pub fn set_voxel(&mut self, pixels: f32) {
		self.voxel = pixels.max(0.05);
	}

	/// Whether the map draws as its own surfaces, the triangles it is made
	/// of, rasterized and culled at the screen's native resolution, instead
	/// of voxelized into cubes; the shell and the trace take over without
	/// it.
	pub fn set_surfaces(&mut self, surfaces: bool) {
		self.surfaces = surfaces;
	}

	/// Whether the map is marched through the shell field, voxelized ahead
	/// of the view, instead of traced per hit.
	pub fn set_shell(&mut self, shell: bool) {
		self.shell = shell;
	}

	/// Whether the shell field's cubes draw their edges pink over their
	/// faces.
	pub fn set_wire(&mut self, wire: bool) {
		self.wire = wire;
	}

	/// Where the shell field's cubes grow: the smallest out to `near` world
	/// units, 0 for where they grow past the voxel setting's pixels, then
	/// `curve` sizes up per doubling of distance.
	pub fn set_lod(&mut self, near: f32, curve: f32) {
		self.lod = world::Lod {
			near: near.max(0.0),
			curve: curve.clamp(0.25, 4.0),
		};
	}

	/// How the baked light's ambient part — sky, bounces and lamps, all but
	/// the straight sun — lands: scaled by `gi`, 1 as baked, and darkened by
	/// the frame's ambient occlusion at strength `ao`, 0 off to 1 full,
	/// looking `radius` world units around each pixel.
	pub fn set_indirect(&mut self, gi: f32, ao: f32, radius: f32) {
		self.indirect = world::Indirect {
			gi: gi.clamp(0.0, 4.0),
			ao: ao.clamp(0.0, 1.0),
			radius: radius.clamp(1.0, 256.0),
			traced: self.indirect.traced,
		};
	}

	/// Whether the sun's shadow is traced per pixel, sharp as the screen,
	/// or read from the bake, as coarse as its texels and cheaper.
	pub fn set_traced_shadows(&mut self, traced: bool) {
		self.indirect.traced = traced;
	}

	/// Fog over the surface view, `density` squared per thousand world
	/// units at height 0 — squared so the low end, where fog is subtle,
	/// gets most of a slider — 0 clear, thinning by e every `height` units
	/// up, its glow tinted by `color`. The voxel views stay clear.
	pub fn set_fog(&mut self, density: f32, height: f32, color: Vec3) {
		let density = density.clamp(0.0, 10.0);
		self.fog = world::Fog {
			density: density * density / 1000.0,
			height: height.clamp(1.0, 1e5),
			color: vec3(
				color.x.clamp(0.0, 4.0),
				color.y.clamp(0.0, 4.0),
				color.z.clamp(0.0, 4.0),
			),
		};
	}

	/// Whether the exposure adapts to the scene, as an eye does: opening
	/// up indoors, closing down in the sun. The brightness setting scales
	/// it either way.
	pub fn set_auto_exposure(&mut self, adapt: bool) {
		self.auto_exposure = adapt;
	}

	/// The range every frame's exposure stays in: the brightness setting
	/// is held to it, and auto exposure adapts within it — `min` the
	/// darkest it closes down to in the sun, `max` the brightest it opens
	/// up to in the dark.
	pub fn set_exposure_range(&mut self, min: f32, max: f32) {
		let min = min.max(0.01);
		self.exposure_range = (min, max.max(min));
	}

	/// A light for this frame only, at `pos`, of linear radiance `color` a
	/// quarter of its `reach` away and nothing past the reach: a muzzle
	/// flash, a lamp the editor placed. Shadowed by the map and by moving
	/// shapes. Past [`world::MAX_LIGHTS`] in a frame the rest are dropped.
	pub fn light(&mut self, pos: Vec3, color: Vec3, reach: f32) {
		self.lights.push(world::DynLight { pos, color, reach });
	}

	/// A moving shape for this frame only, casting shadows in the sun and
	/// every dynamic light: the segment `a` to `b` swept by a ball of
	/// `radius`, a sphere when the two meet. Past [`world::MAX_OCCLUDERS`]
	/// in a frame the rest are dropped.
	pub fn occluder(&mut self, a: Vec3, b: Vec3, radius: f32) {
		self.occluders.push((a, b, radius));
	}

	/// Faces for this frame only, drawn and lit as the map's own — props,
	/// things that move — with `ambient`, the light around them, standing
	/// in for a baked chart. They cast no shadow of their own; give them an
	/// [`Gfx::occluder`] for that.
	pub fn faces(&mut self, faces: &[crate::bsp::Face], ambient: Vec3) {
		scene::loose_corners(faces, ambient, &mut self.loose);
	}

	/// Whether surfaces reflect physically by their finish — specular
	/// highlights, metal, relief — or plain diffuse, as the bake lights
	/// them.
	pub fn set_pbr(&mut self, pbr: bool) {
		self.pbr = pbr;
	}

	/// Whether the frame's dynamic lights light anything.
	pub fn set_dynamic_lights(&mut self, on: bool) {
		self.dynamic_lights = on;
	}

	/// Whether moving shapes cast shadows.
	pub fn set_dynamic_shadows(&mut self, on: bool) {
		self.dynamic_shadows = on;
	}

	/// Whether frames wait for the display's refresh.
	pub fn set_vsync(&mut self, vsync: bool) {
		let mode = if vsync {
			wgpu::PresentMode::AutoVsync
		} else {
			wgpu::PresentMode::AutoNoVsync
		};
		if self.config.present_mode != mode {
			self.config.present_mode = mode;
			self.surface.configure(&self.device, &self.config);
		}
	}

	/// Rays the trace averages per pixel, so cubes and edges finer than a
	/// pixel blend instead of alias: 1, 2 or 4, each costing about as much
	/// trace time again.
	pub fn set_rays(&mut self, rays: f32) {
		self.rays = if rays >= 4.0 {
			4
		} else if rays >= 2.0 {
			2
		} else {
			1
		};
	}

	/// Whether the frame's edges are smoothed after it is drawn (FXAA):
	/// stairs of whole pixels blend into the slopes they stand for, at
	/// the cost of one full-screen pass's reads, whatever draws the map.
	pub fn set_smooth(&mut self, smooth: bool) {
		self.smooth = smooth;
	}

	/// Steps to the next of [`VIEW_MODES`], after the last back to the lit
	/// world.
	pub fn cycle_view_mode(&mut self) {
		self.view_mode = (self.view_mode + 1) % VIEW_MODES.len();
	}

	/// The name of the view the trace shows, from [`VIEW_MODES`].
	pub fn view_mode(&self) -> &'static str {
		VIEW_MODES[self.view_mode]
	}

	/// The playground's spheres: center and radius each, drawn by the trace.
	pub fn set_bodies(&mut self, bodies: &[(Vec3, f32)]) {
		if let Some(world) = &mut self.world {
			world.set_bodies(&self.queue, bodies);
		}
	}

	/// Dents the playground's spheres left in the map: center and radius
	/// each, carved out of the traced matter.
	pub fn set_dents(&mut self, dents: &[(Vec3, f32)]) {
		if let Some(world) = &mut self.world {
			world.set_dents(&self.queue, dents);
		}
	}

	/// An RGBA8 image for [`Gfx::mesh`].
	pub fn texture(&mut self, width: u32, height: u32, rgba: &[u8]) -> TextureId {
		self
			.overlay
			.texture(&self.device, &self.queue, width, height, rgba)
	}

	/// Frees every [`Gfx::texture`]; their ids are invalid afterwards. For a
	/// game that reloads in the open window and uploads its images again.
	pub fn clear_textures(&mut self) {
		self.overlay.clear_textures();
	}

	/// A world-space line, hidden where the map stands in front of it.
	pub fn line_3d(&mut self, from: Vec3, to: Vec3, color: Color) {
		self.overlay.line_3d(from, to, color);
	}

	/// A solid axis-aligned box of `size` around `center`.
	pub fn cube(&mut self, center: Vec3, size: Vec3, color: Color) {
		self.overlay.cube(center, size, color);
	}

	/// Textured triangles drawn over the world, as the viewmodel is: never
	/// clipped by the map, hidden only behind themselves.
	pub fn mesh(&mut self, texture: TextureId, vertices: impl IntoIterator<Item = Vertex>) {
		self.overlay.mesh(texture, vertices);
	}

	/// Boxes drawn like [`Gfx::mesh`], over the world: the stand-in viewmodel.
	pub fn front_cube(&mut self, center: Vec3, size: Vec3, color: Color) {
		self.overlay.front_cube(center, size, color);
	}

	/// A filled rectangle in pixels from the top left.
	pub fn rect(&mut self, x: f32, y: f32, width: f32, height: f32, color: Color) {
		self.overlay.rect(x, y, width, height, color);
	}

	/// A rectangle shaded between its corners' colors: top left, top right,
	/// bottom right, bottom left.
	pub fn gradient(&mut self, x: f32, y: f32, width: f32, height: f32, colors: [Color; 4]) {
		self.overlay.gradient(x, y, width, height, colors);
	}

	/// Cuts the flat drawing that follows — rects, lines, text — to `rect`,
	/// x, y, width, height in pixels; none lets it draw anywhere again. The
	/// frame starts unclipped.
	pub fn set_clip(&mut self, rect: Option<[f32; 4]>) {
		self.overlay.set_clip(rect);
	}

	/// A line `thickness` pixels wide.
	pub fn line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, color: Color) {
		self.overlay.line(x1, y1, x2, y2, thickness, color);
	}

	/// Text `size` pixels tall with its baseline at `y`.
	pub fn text(&mut self, text: &str, x: f32, y: f32, size: f32, color: Color) {
		self.overlay.text(&self.queue, text, x, y, size, color);
	}

	/// How far [`Gfx::text`] advances drawing `text` at `size`.
	pub fn text_width(&mut self, text: &str, size: f32) -> f32 {
		self.overlay.text_width(&self.queue, text, size)
	}

	fn present(&mut self) {
		let frame = match self.surface.get_current_texture() {
			Ok(frame) => frame,
			Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
				self.surface.configure(&self.device, &self.config);
				self.overlay.clear();
				self.lights.clear();
				self.occluders.clear();
				self.loose.clear();
				return;
			}
			Err(_) => {
				self.overlay.clear();
				self.lights.clear();
				self.occluders.clear();
				self.loose.clear();
				return;
			}
		};
		let target = frame.texture.create_view(&Default::default());
		let size = self.size();
		// With a world drawn the frame lands in its own color first, to be
		// shaded by its own depth; without one the sky clears straight to
		// the window.
		let occlude = self.world.is_some() && self.view.is_some();
		let mut encoder = self.device.create_command_encoder(&Default::default());
		if let Some(timer) = &mut self.timer {
			timer.begin(&mut encoder);
		}
		if self.shell && !self.surfaces {
			if let (Some(world), Some(view)) = (&mut self.world, &self.view) {
				world.prepare(&self.queue, &mut encoder, view, size, self.voxel, self.lod);
			}
			mark(&mut self.timer, &mut encoder, "shell");
		}
		let sky = self.sky;
		// The brightness setting, scaled by the eye's adaptation when it
		// adapts.
		let (lo, hi) = self.exposure_range;
		let base = self.exposure.clamp(lo, hi);
		self.meter.set_range(lo / base, hi / base);
		let exposure = base
			* if self.auto_exposure {
				self.meter.scale()
			} else {
				1.0
			};
		if let Some(world) = &mut self.world {
			world.set_loose(&self.device, &self.queue, &self.loose);
			let lights: &[world::DynLight] = if self.dynamic_lights {
				&self.lights
			} else {
				&[]
			};
			world.set_dynamic(
				&self.queue,
				lights,
				&self.occluders,
				self.pbr,
				self.dynamic_shadows,
			);
		}
		{
			let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("world"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: if occlude { &self.scene } else { &target },
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
					view: &self.depth,
					depth_ops: Some(wgpu::Operations {
						load: wgpu::LoadOp::Clear(0.0),
						store: wgpu::StoreOp::Store,
					}),
					stencil_ops: None,
				}),
				timestamp_writes: None,
				occlusion_query_set: None,
			});
			if let (Some(world), Some(view)) = (&mut self.world, &self.view) {
				world.draw(
					&self.queue,
					&mut pass,
					view,
					size,
					sky,
					exposure,
					self.encode,
					self.voxel,
					self.view_mode as u32,
					self.surfaces,
					self.shell,
					self.rays,
					self.wire,
					self.indirect,
					if self.surfaces {
						self.fog
					} else {
						world::Fog {
							density: 0.0,
							..self.fog
						}
					},
					self.smooth,
				);
			}
		}
		mark(&mut self.timer, &mut encoder, "trace");
		if occlude && self.auto_exposure {
			self.meter.measure(&self.device, &mut encoder, &self.scene);
		}
		if let (Some(world), Some(_)) = (&mut self.world, &self.view) {
			world.occlude(
				&self.device,
				&mut encoder,
				&target,
				&self.scene,
				&self.depth,
				(self.config.width, self.config.height),
			);
			mark(&mut self.timer, &mut encoder, "occlusion");
		}
		self.overlay.draw(
			&self.device,
			&self.queue,
			&mut encoder,
			&target,
			&self.depth,
			self.view.as_ref(),
			size,
		);
		mark(&mut self.timer, &mut encoder, "overlay");
		if let Some(timer) = &mut self.timer {
			timer.end(&mut encoder);
		}
		self.queue.submit([encoder.finish()]);
		self
			.meter
			.collect(&self.device, self.auto_exposure && occlude);
		if let Some(timer) = &mut self.timer {
			timer.collect(&self.device);
		}
		self.window.pre_present_notify();
		frame.present();
		self.overlay.clear();
		self.lights.clear();
		self.occluders.clear();
		self.loose.clear();
		self.view = None;
	}
}
