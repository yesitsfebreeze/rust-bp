//! The game's cvars, declared with [`cvar::cvars!`], and the movement variables a
//! tick of `player::Player` reads, gathered from them once per tick.

use crate::player::MovementVars;

#[rustfmt::skip]
engine::cvar::cvars! {
	/// Every tunable value of the game. Fields are read directly by the code that
	/// uses them; the console and config address them by name.
	pub struct Cvars {
		// Defaults are Counter-Strike 1.6's, except where noted.
		sv_gravity = 800.0, "World gravity, units/s².";
		sv_accelerate = 10.0, "Ground acceleration, multiples of wish speed per second; doubled with sv_friction so top speed is unchanged. Not in CS 1.6.";
		sv_airaccelerate = 10.0, "Air acceleration; air wish speed is capped at 30 u/s.";
		sv_friction = 8.0, "Ground friction; doubled from CS 1.6 so the player stops in half the distance. Not in CS 1.6.";
		sv_edgefriction = 2.0, "Friction multiplier near a drop-off.";
		sv_stopspeed = 75.0, "Below this speed friction acts as if moving this fast.";
		sv_maxspeed = 320.0, "Server cap on player speed, u/s.";
		sv_maxvelocity = 2000.0, "Cap on each velocity component, u/s.";
		sv_stepsize = 18.0, "Highest step walked up without jumping, units.";
		sv_stepsmooth = 1.0, "Stick to stairs going down and ease the view through steps; 0 is CS 1.6. Not in CS 1.6.";
		sv_stepangle = 45.0, "Slope in degrees the view follows through a step at walking speed. Not in CS 1.6.";
		sv_tickrate = 128.0, "Movement ticks per second; rendering interpolates between ticks. CS 1.6 ran one tick per frame.";
		sv_infinite_money = 0.0, "Buying never checks or spends money. Not in CS 1.6.";
		player_maxspeed = 250.0, "Speed of the held weapon, u/s (knife 250, AWP 210).";
		cl_forwardspeed = 400.0, "Forward key speed, u/s.";
		cl_backspeed = 400.0, "Back key speed, u/s.";
		cl_sidespeed = 400.0, "Strafe key speed, u/s.";
		cl_upspeed = 320.0, "Noclip up and down speed, u/s.";
		cl_movespeedkey = 0.52, "Speed multiplier while walking (shift).";
		sensitivity = 3.0, "Mouse sensitivity.";
		m_yaw = 0.022, "Degrees of yaw per mouse count, before sensitivity.";
		m_pitch = 0.022, "Degrees of pitch per mouse count, before sensitivity; negative inverts.";
		fov = 90.0, "Horizontal field of view at 4:3, degrees; wider screens see more.";
		cl_interp = 1.0, "Draw the player between the last two ticks; 0 draws the last tick.";
		cl_showspeed = 1.0, "Show horizontal speed.";
		cl_showfps = 1.0, "Show frame rate and tick rate.";
		cl_crosshair = 1.0, "Draw the crosshair.";
		sv_penetration = 1.0, "Bullets punch through thin walls, losing damage per material.";
		r_subsurface = 1.0, "Subsurface scattering strength; 0 renders every surface opaque. Not in CS 1.6.";
		r_exposure = 1.0, "Scene brightness before tone mapping. Not in CS 1.6.";
		r_gi = 1.0, "Strength of the baked bounce and sky light, all but direct sun: 1 as baked, 0 sunlight alone. Not in CS 1.6.";
		r_ao = 1.0, "Screen-space ambient occlusion over the bounce and sky light: 0 off, 1 full. Not in CS 1.6.";
		r_ao_radius = 24.0, "How far around a pixel ambient occlusion looks for matter, units. Not in CS 1.6.";
		r_autoexposure = 1.0, "Adapt brightness to the scene as an eye does: brighter indoors, darker in the sun. Not in CS 1.6.";
		r_exposure_min = 0.25, "Lowest exposure: the Brightness slider's left end, and the darkest auto exposure closes down to in bright scenes. Not in CS 1.6.";
		r_exposure_max = 4.0, "Highest exposure: the Brightness slider's right end, and the brightest auto exposure opens up to in dark scenes. Not in CS 1.6.";
		r_fog = 1.0, "Fog thickness near the ground, squared, per thousand units: 0 clear air, 2 four times 1. Glows with the sky and toward the sun; thins with height. Not in CS 1.6.";
		r_fog_height = 512.0, "Units of height over which the fog thins to about a third: smaller hugs the ground. Not in CS 1.6.";
		r_fog_r = 1.0, "Fog color, red: tints the sky and sun light the fog glows with, 1 untinted. Not in CS 1.6.";
		r_fog_g = 1.0, "Fog color, green: tints the sky and sun light the fog glows with, 1 untinted. Not in CS 1.6.";
		r_fog_b = 1.0, "Fog color, blue: tints the sky and sun light the fog glows with, 1 untinted. Not in CS 1.6.";
		r_shadows = 1.0, "Sharp sun shadow from a shadow map baked at load; 0 reads the lightmap's coarser one. Not in CS 1.6.";
		r_pbr = 1.0, "Physically based surfaces: specular highlights by each surface's roughness and metalness, and relief from its texture; 0 plain diffuse. Not in CS 1.6.";
		r_dlights = 1.0, "Dynamic lights, such as muzzle flashes, lighting the world; only moving things shadow them. Not in CS 1.6.";
		r_dynshadows = 1.0, "Moving things, the player and thrown spheres, cast shadows over the world's. Not in CS 1.6.";
		r_vsync = 0.0, "Wait for the display's refresh: no tearing, frame rate capped at the refresh rate. Not in CS 1.6.";
		r_fxaa = 1.0, "Smooth the stairs along the frame's edges after it is drawn (FXAA), whatever draws the map; costs one full-screen pass. Not in CS 1.6.";
	}
}

/// The movement cvars a tick reads, gathered once per tick.
pub fn movement_vars(c: &Cvars) -> MovementVars {
	MovementVars {
		gravity: c.sv_gravity,
		accelerate: c.sv_accelerate,
		airaccelerate: c.sv_airaccelerate,
		friction: c.sv_friction,
		edgefriction: c.sv_edgefriction,
		stopspeed: c.sv_stopspeed,
		// PM_CheckParameters: a zero client speed means no weapon limit.
		maxspeed: if c.player_maxspeed != 0.0 {
			c.player_maxspeed.min(c.sv_maxspeed)
		} else {
			c.sv_maxspeed
		},
		maxvelocity: c.sv_maxvelocity,
		stepsize: c.sv_stepsize,
		stepsmooth: c.sv_stepsmooth != 0.0,
		stepangle: c.sv_stepangle,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use engine::cvar::Registry;

	#[test]
	fn cvars_round_trip_through_toml() {
		let mut c = Cvars::default();
		c.set("sv_gravity", 600.0).unwrap();
		let mut d = Cvars::default();
		assert_eq!(d.apply_toml(&c.to_toml()), Ok(vec![]));
		assert_eq!(c, d);
	}

	#[test]
	fn movement_vars_caps_weapon_speed_by_the_server() {
		let c = Cvars::default();
		assert_eq!(movement_vars(&c).maxspeed, 250.0);
		let mut c = Cvars {
			sv_maxspeed: 200.0,
			..Default::default()
		};
		assert_eq!(movement_vars(&c).maxspeed, 200.0);
		c.player_maxspeed = 0.0;
		assert_eq!(movement_vars(&c).maxspeed, 200.0);
	}
}
