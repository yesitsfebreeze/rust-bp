/// Material categories shared by engine rendering and gameplay systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
	Tile,
	Concrete,
	Marble,
	Jade,
	Steel,
}

impl Surface {
	pub const ALL: [Surface; 5] = [
		Surface::Tile,
		Surface::Concrete,
		Surface::Marble,
		Surface::Jade,
		Surface::Steel,
	];
}
