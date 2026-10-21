fn conf() -> macroquad::prelude::Conf {
	macroquad::prelude::Conf {
		window_title: "player".into(),
		window_width: 1600,
		window_height: 900,
		sample_count: 4,
		..Default::default()
	}
}

#[macroquad::main(conf)]
async fn main() {
	gameplay::run().await;
}
