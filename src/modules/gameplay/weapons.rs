//! The 1.6 weapon table and the gun in the player's hands. Every number comes
//! from Counter-Strike 1.6's mp.dll, loaded from `assets/weapons/` — one TOML
//! file per weapon, editable without a rebuild — with the built-in files
//! written there first so a fresh checkout has its own.
//!
//! The heat system is 1.6's own: rifles count shots since deploy or reload
//! into a cubic accuracy factor, pistols let accuracy recover over time, and
//! the view kick is mp.dll's KickBack, which grows with consecutive shots and
//! flips its horizontal direction at per-weapon odds.

use engine::gpu::{Color, Gfx};
use engine::linalg::{self, Vec3};

/// The built-in weapon files, written to `assets/weapons/` when missing and
/// the source of the defaults.
const DEFAULTS: &[(&str, &str)] = &[
	("knife", include_str!("../../../assets/weapons/knife.toml")),
	(
		"glock18",
		include_str!("../../../assets/weapons/glock18.toml"),
	),
	("usp", include_str!("../../../assets/weapons/usp.toml")),
	(
		"deagle",
		include_str!("../../../assets/weapons/deagle.toml"),
	),
	("mp5", include_str!("../../../assets/weapons/mp5.toml")),
	("ak47", include_str!("../../../assets/weapons/ak47.toml")),
	("m4a1", include_str!("../../../assets/weapons/m4a1.toml")),
	("awp", include_str!("../../../assets/weapons/awp.toml")),
];

/// The built-in economy file.
const DEFAULT_MONEY: &str = include_str!("../../../assets/money.toml");

/// Which heat model a weapon uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Heat {
	/// Rifles: accuracy = base + shots^power / div, capped. The counter only
	/// resets on deploy and reload, exactly as in 1.6.
	Shots {
		base: f32,
		power: f32,
		div: f32,
		cap: f32,
	},
	/// Pistols: accuracy recovers toward `max` after each shot; spread scales
	/// with the remaining inaccuracy.
	Recover {
		window: f32,
		rate: f32,
		min: f32,
		max: f32,
	},
	/// The AWP: fixed spreads per state, no accuracy factor.
	None,
}

/// One spread state: `mul * factor + add`, with ducking overrides that win
/// while grounded and ducked.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpreadState {
	pub mul: f32,
	pub add: f32,
	pub duck_mul: Option<f32>,
	pub duck_add: Option<f32>,
}

/// The spread states a shot can be in, and the speed boundaries between them.
#[derive(Clone, Debug, Default)]
pub struct Spread {
	pub still_speed: f32,
	pub slow_speed: f32,
	pub air: SpreadState,
	pub still: Option<SpreadState>,
	pub slow: Option<SpreadState>,
	pub fast: Option<SpreadState>,
}

impl Spread {
	/// The state's multiplier pair, ducking overrides winning while grounded.
	fn state(&self, grounded: bool, ducked: bool, speed: f32) -> (f32, f32) {
		if !grounded {
			return (self.air.mul, self.air.add);
		}
		// A weapon with no `still` table falls to its `slow` one, as 1.6's
		// rifles name only slow and fast.
		let s = if speed <= self.still_speed {
			self.still.or(self.slow)
		} else if speed <= self.slow_speed {
			self.slow
		} else {
			self.fast
		};
		s.map(|s| {
			if ducked {
				(s.duck_mul.unwrap_or(s.mul), s.duck_add.unwrap_or(s.add))
			} else {
				(s.mul, s.add)
			}
		})
		.unwrap_or((0.0, 0.0))
	}
}

/// One KickBack state: kick_x, kick_y, growth_x, growth_y, cap_x, cap_y,
/// flip_rand, in mp.dll's order.
#[derive(Clone, Copy, Debug)]
pub struct KickBack(pub [f32; 7]);

/// The glock18's three-round burst mode.
#[derive(Clone, Copy, Debug)]
pub struct Burst {
	pub rounds: u32,
	pub interval: f32,
	pub next: f32,
}

/// The AWP's scope: right-click steps through `fovs`, then out, capping run
/// speed while scoped.
#[derive(Clone, Debug)]
pub struct Zoom {
	pub fovs: Vec<f32>,
	pub scoped_speed: f32,
	pub sound: Option<String>,
}

/// The weapon's sound files, by name under `sound/weapons/`.
#[derive(Clone, Debug, Default)]
pub struct Sounds {
	pub fire: Vec<String>,
	pub deploy: Option<String>,
	pub dry: Option<String>,
	/// Reload sounds at their fraction of the reload, `file@fraction`.
	pub reload: Vec<String>,
	pub stab: Option<String>,
	#[allow(dead_code)]
	pub hit: Option<String>,
	pub wall: Option<String>,
}

/// One weapon's numbers, from Counter-Strike 1.6.
#[derive(Clone, Debug)]
pub struct Weapon {
	pub name: String,
	pub slot: usize,
	pub price: u32,
	pub ammo_price: u32,
	pub ammo_name: String,
	pub damage: f32,
	/// The knife's slash from behind, and the right-click stab from front and
	/// behind; melee only.
	#[allow(dead_code)]
	pub back_damage: f32,
	pub stab_damage: f32,
	#[allow(dead_code)]
	pub stab_back_damage: f32,
	/// The knife's reach, units.
	#[allow(dead_code)]
	pub reach: f32,
	pub melee: bool,
	/// How far a bullet flies, units.
	pub range: f32,
	pub range_mod: f32,
	pub pen: u32,
	pub cycle: f32,
	pub stab_cycle: f32,
	pub auto: bool,
	pub clip: u32,
	pub reserve: u32,
	pub maxspeed: f32,
	/// Seconds the draw takes; the viewmodel's draw sequence length wins when
	/// the model loaded.
	pub deploy: f32,
	pub reload: f32,
	/// A fixed vertical view kick (AWP, pistols: -2 degrees) in place of the
	/// KickBack states.
	pub punch_x: Option<f32>,
	pub heat: Heat,
	pub spread: Spread,
	/// KickBack states, in air / moving / ducked / standing order.
	pub punch: Option<[KickBack; 4]>,
	pub burst: Option<Burst>,
	pub burst_spread: Option<Spread>,
	pub zoom: Option<Zoom>,
	pub sounds: Sounds,
}

impl Weapon {
	/// The view kick state for the shot's situation, if the weapon has them.
	fn kicks(&self, grounded: bool, ducked: bool, moving: bool) -> Option<KickBack> {
		let p = self.punch?;
		Some(match (grounded, ducked, moving) {
			(false, _, _) => p[0],
			(_, true, true) => p[1],
			(_, true, false) => p[2],
			(_, false, _) => p[3],
		})
	}

	/// Fire direction: the punched view through the heat-derived cone, as
	/// GoldSrc's FireBullets mixes two uniform draws per axis into the spread.
	pub fn dir(&self, forward: Vec3, right: Vec3, up: Vec3, spread: f32) -> Vec3 {
		let uniform = || crate::rng::range(-0.5, 0.5) + crate::rng::range(-0.5, 0.5);
		(forward + right * (uniform() * spread) + up * (uniform() * spread)).normalize()
	}
}

/// What a tick of the clock did.
#[derive(Default)]
pub struct Tick {
	/// The reload sounds whose moment arrived, by their `@fraction`.
	pub reload_sounds: Vec<f32>,
	/// A burst round fires now, without the trigger.
	pub burst: bool,
}

/// The gun in the player's hands: its loaded state, heat and view kick.
pub struct Armed {
	pub clip: u32,
	pub reserve: u32,
	/// Shots since deploy or reload; the rifle heat counter.
	pub shots: u32,
	/// The heat factor; rifles recompute it per shot, pistols let it recover.
	pub accuracy: f32,
	last_fire: Option<f32>,
	pub next_fire: f32,
	/// View kick in degrees, pitch and yaw; decayed every frame by the caller.
	pub punch: (f32, f32),
	punch_flip: bool,
	/// The reload's start and end, and its staged sounds still to play.
	pub reload_at: Option<(f32, f32)>,
	reload_sounds: Vec<(f32, f32)>,
	/// Deploy's end time; no firing before it.
	pub deploy_end: f32,
	/// The glock18's burst mode, and a burst in flight: rounds left and the
	/// next one's time.
	pub burst_mode: bool,
	pub burst_left: u32,
	pub burst_next: f32,
	/// The knife's right-click armed a stab: the next fire stabs.
	pub stab: bool,
	/// Zoom level: 0 unscoped, 1 then 2 through the weapon's fovs.
	pub zoom: u8,
}

impl Armed {
	pub fn new(w: &Weapon, now: f32) -> Armed {
		Armed {
			clip: w.clip,
			reserve: w.reserve,
			shots: 0,
			accuracy: Armed::rest_accuracy(w.heat),
			last_fire: None,
			next_fire: now,
			punch: (0.0, 0.0),
			punch_flip: false,
			reload_at: None,
			reload_sounds: Vec::new(),
			deploy_end: now + w.deploy,
			burst_mode: false,
			burst_left: 0,
			burst_next: 0.0,
			stab: false,
			zoom: 0,
		}
	}

	/// Where a weapon's accuracy rests between strings of fire.
	fn rest_accuracy(h: Heat) -> f32 {
		match h {
			Heat::Recover { max, .. } => max,
			Heat::Shots { base, .. } => base,
			Heat::None => 0.0,
		}
	}

	/// Switches to another weapon, drawing it; the new weapon's loaded state
	/// comes with it, the old one saved by the caller.
	pub fn switch(&mut self, w: &Weapon, clip: u32, reserve: u32, now: f32) {
		*self = Armed {
			clip,
			reserve,
			deploy_end: now + w.deploy,
			..Armed::new(w, now)
		};
	}

	/// Reloads: ends `w.reload` later, stages the sounds, resets the heat.
	pub fn start_reload(&mut self, w: &Weapon, now: f32) {
		if self.reload_at.is_some() || self.clip == w.clip || self.reserve == 0 {
			return;
		}
		self.reload_at = Some((now, now + w.reload));
		self.reload_sounds = w
			.sounds
			.reload
			.iter()
			.map(|s| (w.reload * fraction(s), 0.0))
			.collect();
		self.shots = 0;
		self.accuracy = Armed::rest_accuracy(w.heat);
		self.burst_left = 0;
		self.zoom = 0;
	}

	/// The clock's work: staged reload sounds, the reload's finish, scheduled
	/// burst rounds. The caller plays what the tick reports.
	pub fn tick(&mut self, w: &Weapon, now: f32) -> Tick {
		let mut tick = Tick::default();
		if let Some((start, end)) = self.reload_at {
			for (at, played) in &mut self.reload_sounds {
				if *played == 0.0 && now >= start + *at {
					*played = 1.0;
					tick.reload_sounds.push(*at);
				}
			}
			if now >= end {
				let want = w.clip.min(self.clip + self.reserve);
				self.reserve -= want - self.clip;
				self.clip = want;
				self.reload_at = None;
			}
		}
		if self.burst_left > 0 && now >= self.burst_next {
			tick.burst = true;
		}
		tick
	}

	/// Whether the trigger can pull now.
	pub fn can_fire(&self, now: f32) -> bool {
		now >= self.next_fire && self.reload_at.is_none() && now >= self.deploy_end
	}

	/// The spread for a shot in this situation, from the heat factor and state.
	pub fn spread(&self, w: &Weapon, grounded: bool, ducked: bool, speed: f32) -> f32 {
		let table = if self.burst_mode || self.burst_left > 0 {
			w.burst_spread.as_ref().unwrap_or(&w.spread)
		} else {
			&w.spread
		};
		let (mul, add) = table.state(grounded, ducked, speed);
		let factor = match w.heat {
			Heat::Shots { .. } => self.accuracy,
			Heat::Recover { .. } => 1.0 - self.accuracy,
			Heat::None => 1.0,
		};
		mul * factor + add
	}

	/// One shot's bookkeeping, minus the bullet itself: clip, heat, view kick.
	/// Returns the shot's spread, or None when the trigger cannot pull or the
	/// gun is empty (which schedules the dry click).
	pub fn fire(
		&mut self,
		w: &Weapon,
		now: f32,
		grounded: bool,
		ducked: bool,
		speed: f32,
	) -> Option<f32> {
		let burst_round = self.burst_left > 0 && now >= self.burst_next;
		if !burst_round && !self.can_fire(now) {
			return None;
		}
		if self.clip == 0 && !w.melee {
			// The dry click schedules itself, and a burst in flight dies.
			self.burst_left = 0;
			self.next_fire = now + 0.2;
			return None;
		}
		if self.clip > 0 {
			self.clip -= 1;
		}
		self.shots += 1;
		// The heat counts this shot too: 1.6's first round already fires at
		// `base + 1/div`, so the accuracy updates before the spread reads it.
		match w.heat {
			Heat::Shots {
				base,
				power,
				div,
				cap,
			} => {
				self.accuracy = (base + (self.shots as f32).powf(power) / div).min(cap);
			}
			Heat::Recover {
				window,
				rate,
				min,
				max,
			} => {
				if let Some(last) = self.last_fire {
					self.accuracy = (self.accuracy - (window - (now - last)) * rate).clamp(min, max);
				}
				self.last_fire = Some(now);
			}
			Heat::None => {}
		}
		let spread = self.spread(w, grounded, ducked, speed);
		// The burst schedules its remaining rounds and its longer cycle from
		// the first one.
		if let Some(b) = w.burst {
			if burst_round {
				self.burst_left -= 1;
				self.burst_next = now + b.interval;
			} else if self.burst_mode {
				self.burst_left = b.rounds - 1;
				self.burst_next = now + b.interval;
				self.next_fire = now + b.next;
			}
		} else if !burst_round {
			self.next_fire = now + w.cycle;
		}
		// The view kick: mp.dll's KickBack, or the fixed pistol kick.
		if let Some(k) = w.kicks(grounded, ducked, speed > 0.0) {
			self.kick(&k.0);
		} else if let Some(x) = w.punch_x {
			self.punch.0 += x;
		}
		Some(spread)
	}

	/// The knife's secondary fire arms a stab: the next fire this frame stabs
	/// on the stab's own cycle.
	pub fn want_stab(&mut self, w: &Weapon, now: f32) -> bool {
		if !w.melee || !self.can_fire(now) {
			return false;
		}
		self.stab = true;
		true
	}

	/// The glock18's secondary fire toggles the burst mode.
	pub fn toggle_burst(&mut self, w: &Weapon, now: f32) -> bool {
		if w.burst.is_none() || self.reload_at.is_some() || now < self.deploy_end {
			return false;
		}
		self.burst_mode = !self.burst_mode;
		self.next_fire = self.next_fire.max(now + 0.3);
		true
	}

	/// The AWP's secondary fire steps the scope through its fovs.
	pub fn toggle_zoom(&mut self, w: &Weapon, now: f32) -> Option<f32> {
		let z = w.zoom.as_ref()?;
		if self.reload_at.is_some() || now < self.deploy_end {
			return None;
		}
		self.zoom = (self.zoom + 1) % (z.fovs.len() as u8 + 1);
		self.next_fire = self.next_fire.max(now + 0.3);
		Some(self.fov(w))
	}

	/// The fov to render with, 0 when unzoomed.
	pub fn fov(&self, w: &Weapon) -> f32 {
		if self.zoom == 0 {
			0.0
		} else {
			w.zoom.as_ref().unwrap().fovs[(self.zoom - 1) as usize]
		}
	}

	/// mp.dll's KickBack: vertical kick grows with consecutive shots and clamps
	/// at its cap; the horizontal kick flips direction at the flip odds.
	fn kick(&mut self, k: &[f32; 7]) {
		let (mut kx, mut ky) = (k[0], k[1]);
		if self.shots != 1 {
			kx += self.shots as f32 * k[2];
			ky += self.shots as f32 * k[3];
		}
		self.punch.0 = (self.punch.0 - kx).max(-k[4]);
		if (crate::rng::range(0.0, k[6] + 1.0) as i32) == 0 {
			self.punch_flip = !self.punch_flip;
		}
		let sy = if self.punch_flip { ky } else { -ky };
		self.punch.1 = (self.punch.1 + sy).clamp(-k[5], k[5]);
	}
}

/// The `@fraction` part of a staged reload sound, 0 when there is none.
fn fraction(s: &str) -> f32 {
	s.rsplit('@')
		.next()
		.and_then(|f| f.parse().ok())
		.unwrap_or(0.0)
}

// --- TOML loading -----------------------------------------------------------

/// Reads a scalar of any numeric kind.
fn num(t: &toml::Table, key: &str) -> Option<f32> {
	t.get(key).and_then(|v| match v {
		toml::Value::Float(f) => Some(*f as f32),
		toml::Value::Integer(i) => Some(*i as f32),
		_ => None,
	})
}

fn string(t: &toml::Table, key: &str) -> Option<String> {
	t.get(key).and_then(|v| v.as_str().map(String::from))
}

fn flag(t: &toml::Table, key: &str) -> bool {
	t.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn u32_of(t: &toml::Table, key: &str) -> Option<u32> {
	t.get(key).and_then(|v| match v {
		toml::Value::Integer(i) => Some(*i as u32),
		_ => None,
	})
}

fn heat(t: &toml::Table) -> Heat {
	let Some(h) = t.get("heat").and_then(|v| v.as_table()) else {
		return Heat::None;
	};
	match h.get("kind").and_then(|v| v.as_str()) {
		Some("shots") => Heat::Shots {
			base: num(h, "base").unwrap_or(0.0),
			power: num(h, "power").unwrap_or(2.0),
			div: num(h, "div").unwrap_or(200.0),
			cap: num(h, "cap").unwrap_or(1.0),
		},
		Some("recover") => Heat::Recover {
			window: num(h, "window").unwrap_or(0.3),
			rate: num(h, "rate").unwrap_or(0.275),
			min: num(h, "min").unwrap_or(0.6),
			max: num(h, "max").unwrap_or(0.9),
		},
		_ => Heat::None,
	}
}

fn spread_state(t: &toml::Table) -> SpreadState {
	SpreadState {
		mul: num(t, "mul").unwrap_or(0.0),
		add: num(t, "add").unwrap_or(0.0),
		duck_mul: num(t, "duck_mul"),
		duck_add: num(t, "duck_add"),
	}
}

fn spread(t: &toml::Table, key: &str) -> Spread {
	let empty = toml::Table::new();
	let s = t.get(key).and_then(|v| v.as_table()).unwrap_or(&empty);
	let state = |name: &str| s.get(name).and_then(|v| v.as_table()).map(spread_state);
	Spread {
		still_speed: num(s, "still_speed").unwrap_or(0.0),
		slow_speed: num(s, "slow_speed").unwrap_or(0.0),
		air: state("air").unwrap_or_default(),
		still: state("still"),
		slow: state("slow"),
		fast: state("fast"),
	}
}

fn kicks(t: &toml::Table) -> Option<[KickBack; 4]> {
	let p = t.get("punch")?.as_table()?;
	let one = |name: &str| {
		let kick = p.get(name)?.as_table()?.get("kick")?.as_array()?;
		let mut k = [0.0f32; 7];
		for (i, v) in kick.iter().take(7).enumerate() {
			k[i] = num(&kick_table(v), "kick").unwrap_or(0.0);
		}
		Some(KickBack(k))
	};
	Some([one("air")?, one("move")?, one("duck")?, one("stand")?])
}

/// Wraps an array element the shared `num` helper can read.
fn kick_table(v: &toml::Value) -> toml::Table {
	let mut t = toml::Table::new();
	t.insert("kick".into(), v.clone());
	t
}

fn one_weapon(t: &toml::Table) -> Weapon {
	Weapon {
		name: string(t, "name").unwrap_or_default(),
		slot: u32_of(t, "slot").unwrap_or(0) as usize,
		price: u32_of(t, "price").unwrap_or(0),
		ammo_price: u32_of(t, "ammo_price").unwrap_or(0),
		ammo_name: string(t, "ammo_name").unwrap_or_default(),
		damage: num(t, "damage").unwrap_or(0.0),
		back_damage: num(t, "back_damage").unwrap_or(0.0),
		stab_damage: num(t, "stab_damage").unwrap_or(0.0),
		stab_back_damage: num(t, "stab_back_damage").unwrap_or(0.0),
		reach: num(t, "reach").unwrap_or(0.0),
		melee: flag(t, "melee"),
		range: num(t, "range").unwrap_or(8192.0),
		range_mod: num(t, "range_mod").unwrap_or(1.0),
		pen: u32_of(t, "pen").unwrap_or(0),
		cycle: num(t, "cycle").unwrap_or(0.1),
		stab_cycle: num(t, "stab_cycle").unwrap_or(0.0),
		auto: flag(t, "auto"),
		clip: u32_of(t, "clip").unwrap_or(0),
		reserve: u32_of(t, "reserve").unwrap_or(0),
		maxspeed: num(t, "maxspeed").unwrap_or(250.0),
		deploy: num(t, "deploy").unwrap_or(1.0),
		reload: num(t, "reload").unwrap_or(0.0),
		punch_x: num(t, "punch_x"),
		heat: heat(t),
		spread: spread(t, "spread"),
		punch: kicks(t),
		burst: t.get("burst").and_then(|v| v.as_table()).map(|b| Burst {
			rounds: u32_of(b, "rounds").unwrap_or(3),
			interval: num(b, "interval").unwrap_or(0.1),
			next: num(b, "next").unwrap_or(0.5),
		}),
		burst_spread: t
			.get("burst_spread")
			.and_then(|v| v.as_table())
			.map(|_| spread(t, "burst_spread")),
		zoom: t.get("zoom").and_then(|v| v.as_table()).map(|z| Zoom {
			fovs: z
				.get("fovs")
				.and_then(|v| v.as_array())
				.map(|a| a.iter().filter_map(num_value).collect())
				.unwrap_or_default(),
			scoped_speed: num(z, "scoped_speed").unwrap_or(150.0),
			sound: string(z, "sound"),
		}),
		sounds: Sounds {
			fire: ["fire1", "fire2"]
				.iter()
				.filter_map(|k| string(t, k))
				.collect(),
			deploy: string(t, "deploy_sound"),
			dry: string(t, "dry_sound"),
			reload: t
				.get("reload_sounds")
				.and_then(|v| v.as_array())
				.map(|a| {
					a.iter()
						.filter_map(|v| v.as_str().map(String::from))
						.collect()
				})
				.unwrap_or_default(),
			stab: string(t, "stab_sound"),
			hit: string(t, "hit_sound"),
			wall: string(t, "wall_sound"),
		},
	}
}

/// Reads a bare array element as a number.
fn num_value(v: &toml::Value) -> Option<f32> {
	match v {
		toml::Value::Float(f) => Some(*f as f32),
		toml::Value::Integer(i) => Some(*i as f32),
		_ => None,
	}
}

/// The whole table: weapons sorted by slot, and the economy.
pub struct Table {
	pub weapons: Vec<Weapon>,
	/// (start, cap, kill award) from `assets/money.toml`.
	pub money: (u32, u32, u32),
}

/// Parses one weapon's TOML file: the fields at the top level, the heat,
/// spread, punch, burst and zoom tables below them.
pub fn parse(text: &str) -> Result<Weapon, String> {
	let doc: toml::Table = text
		.parse()
		.map_err(|e: toml::de::Error| e.message().to_owned())?;
	Ok(one_weapon(&doc))
}

/// The economy values from `money.toml`: start, cap, kill award.
fn economy(text: &str) -> (u32, u32, u32) {
	let doc: toml::Table = text.parse().unwrap_or_default();
	let m = |k: &str, d: u32| {
		doc
			.get(k)
			.and_then(|v| v.as_integer())
			.map(|i| i as u32)
			.unwrap_or(d)
	};
	(m("start", 800), m("cap", 16000), m("kill", 300))
}

/// The default table, as a fresh checkout gets written to disk.
pub fn defaults() -> Result<Table, String> {
	let mut weapons = DEFAULTS
		.iter()
		.map(|(_, text)| parse(text))
		.collect::<Result<Vec<_>, _>>()?;
	weapons.sort_by_key(|w| w.slot);
	Ok(Table {
		weapons,
		money: economy(DEFAULT_MONEY),
	})
}

/// Loads `assets/weapons/*.toml`, one file per weapon, writing the built-in
/// ones there first when missing; `report` collects what happened.
pub fn load(report: &mut Vec<String>) -> Table {
	let dir = crate::root().join("assets").join("weapons");
	for (name, text) in DEFAULTS {
		let path = dir.join(format!("{name}.toml"));
		if !path.exists() {
			match std::fs::write(&path, text) {
				Ok(()) => report.push(format!("wrote defaults to {}", path.display())),
				Err(e) => report.push(format!("{}: {e}", path.display())),
			}
		}
	}
	let mut weapons = Vec::new();
	if let Ok(entries) = std::fs::read_dir(&dir) {
		for entry in entries.flatten() {
			let path = entry.path();
			if path.extension().and_then(|e| e.to_str()) != Some("toml") {
				continue;
			}
			match std::fs::read_to_string(&path)
				.map_err(|e| e.to_string())
				.and_then(|t| parse(&t))
			{
				Ok(w) => weapons.push(w),
				Err(e) => report.push(format!("{}: {e}; skipped", path.display())),
			}
		}
	}
	weapons.sort_by_key(|w| w.slot);
	if weapons.is_empty() {
		report.push("no weapon files loaded; using the built-in defaults".into());
		return defaults().unwrap();
	}
	report.push(format!("{} weapons from {}", weapons.len(), dir.display()));
	// The economy rides in its own file beside the weapons.
	let money_path = crate::root().join("assets").join("money.toml");
	if !money_path.exists() {
		if let Err(e) = std::fs::write(&money_path, DEFAULT_MONEY) {
			report.push(format!("{}: {e}", money_path.display()));
		}
	}
	let money = std::fs::read_to_string(&money_path)
		.map(|t| economy(&t))
		.unwrap_or_else(|_| economy(DEFAULT_MONEY));
	Table { weapons, money }
}

/// A gunshot for the weapon as a WAV file in memory: white noise through a
/// one-pole low-pass with an exponential decay, for when the 1.6 sounds are not
/// installed. Short and bright reads as a pistol, long and dark as a rifle.
pub fn synth_wav(w: &Weapon) -> Vec<u8> {
	let (tau, brightness, gain, dur) = if w.melee {
		(0.04, 0.2, 0.2, 0.12)
	} else if w.cycle < 0.16 {
		(0.06, 0.85, 0.6, 0.12)
	} else {
		(0.12, 0.45, 0.9, 0.25)
	};
	let rate = 44100u32;
	let n = (rate as f32 * dur) as usize;
	let mut samples = vec![0i16; n];
	let mut low = 0.0f32;
	for (i, s) in samples.iter_mut().enumerate() {
		let t = i as f32 / rate as f32;
		let noise = crate::rng::range(-1.0, 1.0);
		low += brightness * (noise - low);
		let v = low * gain * (-t / tau).exp();
		*s = (v.clamp(-1.0, 1.0) * 32767.0) as i16;
	}
	let data_len = (n * 2) as u32;
	let mut wav = Vec::with_capacity(44 + n * 2);
	wav.extend(b"RIFF");
	wav.extend((36 + data_len).to_le_bytes());
	wav.extend(b"WAVEfmt ");
	wav.extend(16u32.to_le_bytes());
	wav.extend(1u16.to_le_bytes()); // PCM
	wav.extend(1u16.to_le_bytes()); // mono
	wav.extend(rate.to_le_bytes());
	wav.extend((rate * 2).to_le_bytes());
	wav.extend(2u16.to_le_bytes());
	wav.extend(16u16.to_le_bytes());
	wav.extend(b"data");
	wav.extend(data_len.to_le_bytes());
	wav.extend(samples.iter().flat_map(|s| s.to_le_bytes()));
	wav
}

/// Loads one sound file from the install's `sound/weapons` directory, falling
/// back to the weapon's synthesized report; the mixer resamples whatever
/// rate the file carries.
pub fn load_clip(dir: Option<&std::path::Path>, file: &str, w: &Weapon) -> audio::Clip {
	if let Some(bytes) = dir.and_then(|d| std::fs::read(d.join(file)).ok()) {
		if let Ok(clip) = audio::Clip::from_wav(&bytes, audio::RATE) {
			return clip;
		}
	}
	audio::Clip::from_wav(&synth_wav(w), audio::RATE).expect("synth wav")
}

/// The weapon in the player's hands, drawn as a plain viewmodel in front of
/// the eye when its GoldSrc model did not load: a body, a barrel and a grip,
/// kicked back by `recoil` (1 at the shot, eased off by the caller).
pub fn draw_viewmodel(
	gfx: &mut Gfx,
	w: &Weapon,
	eye: Vec3,
	forward: Vec3,
	right: Vec3,
	up: Vec3,
	recoil: f32,
) {
	// Slot index reads as size: pistols short, rifles long, the AWP longest.
	let long = match w.slot {
		1 => 0.0,
		2..=4 => 2.0,
		5..=7 => 5.0,
		_ => 6.0,
	};
	// The gun rides low-right, and the shot shoves it toward the eye.
	let base = eye + forward * (8.0 - recoil * 1.5) - right * 3.0 + up * (-2.5 + recoil * 0.7);
	let steel = Color::new(0.16, 0.17, 0.18, 1.0);
	let dark = Color::new(0.1, 0.1, 0.11, 1.0);
	if long == 0.0 {
		// A knife: blade ahead of the fist.
		gfx.front_cube(base, linalg::vec3(0.6, 1.4, 2.2), steel);
		gfx.front_cube(
			base - forward * 1.1 + up * 0.4,
			linalg::vec3(1.0, 1.0, 1.6),
			dark,
		);
		return;
	}
	gfx.front_cube(base, linalg::vec3(1.2, 1.4, 3.0 + long), steel);
	gfx.front_cube(
		base + forward * (1.5 + long * 0.6) - up * 0.2,
		linalg::vec3(0.5, 0.5, 2.0 + long),
		dark,
	);
	gfx.front_cube(
		base - forward * 1.8 + up * 0.9,
		linalg::vec3(0.8, 2.0, 1.0),
		dark,
	);
}

/// The state carried across a module swap: every field packed by hand, since
/// the private ones cannot be reached and a raw copy would dangle through
/// `reload_sounds`' allocation with the old generation's heap.
impl Armed {
	pub fn save(&self) -> Vec<u8> {
		let mut p = crate::state::Packer::new();
		p.u32(self.clip);
		p.u32(self.reserve);
		p.u32(self.shots);
		p.f32(self.accuracy);
		p.bool(self.last_fire.is_some());
		if let Some(t) = self.last_fire {
			p.f32(t);
		}
		p.f32(self.next_fire);
		p.f32(self.punch.0);
		p.f32(self.punch.1);
		p.bool(self.punch_flip);
		p.bool(self.reload_at.is_some());
		if let Some((start, end)) = self.reload_at {
			p.f32(start);
			p.f32(end);
		}
		p.u32(self.reload_sounds.len() as u32);
		for (at, played) in &self.reload_sounds {
			p.f32(*at);
			p.f32(*played);
		}
		p.f32(self.deploy_end);
		p.bool(self.burst_mode);
		p.u32(self.burst_left);
		p.f32(self.burst_next);
		p.bool(self.stab);
		p.u8(self.zoom);
		p.finish()
	}

	pub fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
		let mut u = crate::state::Unpacker::new(bytes)?;
		self.clip = u.u32()?;
		self.reserve = u.u32()?;
		self.shots = u.u32()?;
		self.accuracy = u.f32()?;
		self.last_fire = if u.bool()? { Some(u.f32()?) } else { None };
		self.next_fire = u.f32()?;
		self.punch = (u.f32()?, u.f32()?);
		self.punch_flip = u.bool()?;
		self.reload_at = if u.bool()? {
			Some((u.f32()?, u.f32()?))
		} else {
			None
		};
		let sounds = u.u32()? as usize;
		self.reload_sounds.clear();
		for _ in 0..sounds {
			self.reload_sounds.push((u.f32()?, u.f32()?));
		}
		self.deploy_end = u.f32()?;
		self.burst_mode = u.bool()?;
		self.burst_left = u.u32()?;
		self.burst_next = u.f32()?;
		self.stab = u.bool()?;
		self.zoom = u.u8()?;
		u.end()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn table() -> Table {
		defaults().expect("weapons.toml parses")
	}

	fn by_name<'a>(t: &'a Table, name: &str) -> &'a Weapon {
		t.weapons.iter().find(|w| w.name == name).unwrap()
	}

	#[test]
	fn the_table_loads_and_covers_the_slots() {
		let t = table();
		assert_eq!(t.money, (800, 16000, 300));
		assert_eq!(t.weapons.len(), 8);
		for (slot, w) in t.weapons.iter().enumerate() {
			assert_eq!(w.slot, slot + 1, "{}", w.name);
			assert!(w.damage > 0.0, "{}", w.name);
			assert!(w.cycle >= 0.05, "{} fires no faster than 20/s", w.name);
			assert!(
				w.range_mod > 0.0 && w.range_mod <= 1.0,
				"{} range modifier only loses damage",
				w.name
			);
			assert!(w.maxspeed > 0.0, "{}", w.name);
		}
	}

	#[test]
	fn the_16_values_survived_the_extraction() {
		let t = table();
		let ak = by_name(&t, "ak47");
		assert_eq!(
			(ak.price, ak.damage, ak.cycle, ak.reload),
			(2500, 36.0, 0.0955, 2.45)
		);
		let Heat::Shots {
			base,
			power,
			div,
			cap,
		} = ak.heat
		else {
			panic!("ak47 heat");
		};
		assert_eq!((base, power, div, cap), (0.35, 3.0, 200.0, 1.25));
		let awp = by_name(&t, "awp");
		assert_eq!(
			(awp.price, awp.damage, awp.pen, awp.cycle, awp.maxspeed),
			(4750, 115.0, 2, 1.45, 210.0)
		);
		assert_eq!(awp.punch_x, Some(-2.0));
		let mp5 = by_name(&t, "mp5");
		assert_eq!((mp5.cycle, mp5.range_mod, mp5.reload), (0.075, 0.84, 2.63));
		let deagle = by_name(&t, "deagle");
		assert_eq!(
			(deagle.range, deagle.cycle, deagle.reload),
			(4096.0, 0.225, 2.2)
		);
		let glock = by_name(&t, "glock18");
		assert_eq!(
			glock.burst.map(|b| (b.rounds, b.interval, b.next)),
			Some((3, 0.1, 0.5))
		);
		assert!(glock.burst_spread.is_some());
	}

	#[test]
	fn ak_heat_grows_and_caps_like_16() {
		let t = table();
		let ak = by_name(&t, "ak47").clone();
		let Heat::Shots {
			base,
			power,
			div,
			cap,
		} = ak.heat
		else {
			unreachable!()
		};
		let acc = |shots: u32| (base + (shots as f32).powf(power) / div).min(cap);
		assert!((acc(1) - 0.355).abs() < 1e-4);
		assert!((acc(5) - 0.975).abs() < 1e-4);
		assert_eq!(acc(50), cap);
		// The first shot's standing spread: 0.355 * 0.0275.
		let mut a = Armed::new(&ak, 0.0);
		let s = a.fire(&ak, 2.0, true, false, 0.0).expect("fires");
		assert!((s - 0.355 * 0.0275).abs() < 1e-4, "first shot spread {s}");
		assert_eq!(a.clip, 29);
		assert!((a.next_fire - 2.0 - ak.cycle).abs() < 1e-4);
	}

	#[test]
	fn the_punch_grows_with_shots_and_flips() {
		let t = table();
		let ak = by_name(&t, "ak47").clone();
		let mut a = Armed::new(&ak, 0.0);
		for i in 0..7 {
			a.fire(&ak, 2.0 + i as f32 * 0.1, true, false, 0.0);
		}
		assert!(
			a.punch.0 < -5.0,
			"seven shots kick the pitch: {:?}",
			a.punch
		);
		assert!(a.punch.0 >= -9.0, "the cap holds: {:?}", a.punch);
		assert!(a.punch.1.abs() <= 6.0, "yaw capped: {:?}", a.punch);
	}

	#[test]
	fn mp5_is_ground_forgiving_and_awp_is_not() {
		let t = table();
		let mp5 = by_name(&t, "mp5").clone();
		let mut air = Armed::new(&mp5, 0.0);
		let mut run = Armed::new(&mp5, 0.0);
		for i in 0..10 {
			air.fire(&mp5, 2.0 + i as f32 * 0.1, false, false, 0.0); // airborne
			run.fire(&mp5, 2.0 + i as f32 * 0.1, true, false, 400.0); // running
		}
		let air = air.spread(&mp5, false, false, 0.0);
		let run = run.spread(&mp5, true, false, 400.0);
		// Airborne is acc * 0.2, running on the ground still acc * 0.04.
		assert!(air > run * 3.0, "air {air} vs run {run}");
		let awp = by_name(&t, "awp").clone();
		let mut w = Armed::new(&awp, 0.0);
		let still = w.fire(&awp, 2.0, true, false, 0.0).unwrap();
		assert!(still < 0.01, "still awp {still}");
		let moving = w.fire(&awp, 4.0, true, false, 300.0).unwrap();
		assert!((moving - 0.25).abs() < 1e-3, "running awp {moving}");
	}

	#[test]
	fn reload_transfers_ammo_and_resets_heat() {
		let t = table();
		let ak = by_name(&t, "ak47").clone();
		let mut a = Armed::new(&ak, 0.0);
		for i in 0..5 {
			a.fire(&ak, 2.0 + i as f32 * 0.1, true, false, 0.0);
		}
		a.start_reload(&ak, 3.0);
		assert_eq!(a.shots, 0);
		for i in 0..40 {
			let tick = a.tick(&ak, 3.05 + i as f32 * 0.1);
			assert!(!tick.burst);
		}
		assert_eq!((a.clip, a.reserve), (30, 85));
		assert!(a.reload_at.is_none());
		// Five shots plus 25 fresh ones: all of them played.
		assert_eq!(a.tick(&ak, 9.0).reload_sounds.len(), 0);
	}

	#[test]
	fn dry_fire_schedules_the_click() {
		let t = table();
		let ak = by_name(&t, "ak47").clone();
		let mut a = Armed::new(&ak, 0.0);
		a.clip = 0;
		assert!(a.fire(&ak, 2.0, true, false, 0.0).is_none());
		assert_eq!(a.next_fire, 2.2);
	}

	#[test]
	fn the_glock_burst_fires_three_on_a_long_cycle() {
		let t = table();
		let glock = by_name(&t, "glock18").clone();
		let mut a = Armed::new(&glock, 0.0);
		assert!(a.toggle_burst(&glock, 1.5));
		a.fire(&glock, 2.0, true, false, 0.0);
		assert_eq!((a.burst_left, a.next_fire), (2, 2.5));
		let mut fired = 1;
		for i in 0..50 {
			let at = 2.05 + i as f32 * 0.01;
			if a.tick(&glock, at).burst {
				a.fire(&glock, at, true, false, 0.0);
				fired += 1;
			}
		}
		assert_eq!(fired, 3, "the burst rounds fire on schedule");
		assert_eq!(a.burst_left, 0);
		assert_eq!(a.clip, glock.clip - 3);
	}

	#[test]
	fn zoom_steps_through_the_fovs() {
		let t = table();
		let awp = by_name(&t, "awp").clone();
		let mut a = Armed::new(&awp, 0.0);
		assert_eq!(a.toggle_zoom(&awp, 2.0), Some(40.0));
		assert_eq!(a.toggle_zoom(&awp, 2.1), Some(10.0));
		assert_eq!(a.toggle_zoom(&awp, 2.2), Some(0.0));
	}

	#[test]
	fn synth_wav_stays_a_wav() {
		let t = table();
		for w in &t.weapons {
			let wav = synth_wav(w);
			assert_eq!(&wav[..4], b"RIFF", "{} wav header", w.name);
			assert_eq!(wav.len() % 2, 0, "{} wav is samples", w.name);
		}
	}
}
