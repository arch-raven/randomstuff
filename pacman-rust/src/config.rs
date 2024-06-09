use std::ops::Add;

pub const HEIGHT: usize = 21;
pub const WIDTH: usize = 21;
pub const PALLETE_POINTS: u32 = 20;
pub const POWER_PALLETE_POINTS: u32 = 100;
pub const FRIGHTENED_MODE_TIMER: u32 = 10;
pub const OVERLORD_TEAM_SIZE: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct Coordinates {
    pub x: i32,
    pub y: i32,
}

impl Coordinates {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn to_tuple(&self) -> (usize, usize) {
        (self.x as usize, self.y as usize)
    }
}

impl PartialEq for Coordinates {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Add for Coordinates {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: (WIDTH as i32 + self.x + other.x) % WIDTH as i32,
            y: (HEIGHT as i32 + self.y + other.y) % HEIGHT as i32,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icons {
    Pacman,
    Ghost,
    Wall,
    Pellet,
    PowerPellet,
    Empty,
    Rand,
}

impl Icons {
    pub fn to_char(&self) -> char {
        match self {
            Icons::Pacman => '@',
            Icons::Ghost => '!',
            Icons::Wall => '#',
            Icons::Pellet => '.',
            Icons::PowerPellet => '$',
            Icons::Empty => '-',
            Icons::Rand => 'G',
        }
    }

    pub fn from_char(c: char) -> Self {
        match c {
            '@' => Icons::Pacman,
            '!' => Icons::Ghost,
            '#' => Icons::Wall,
            '.' => Icons::Pellet,
            '$' => Icons::PowerPellet,
            '-' => Icons::Empty,
            _ => Icons::Rand,
        }
    }
}

pub const GRID_LAYOUT: &str = "-###################--#........#........#--#$##.###.#.###.##$#--#.##.#.#####.#.##.#--#....#...#...#....#--####.###-#-###.####-----#.#---!---#.#----#####.#-##=##-#.#####-----.--#!!!#--.-----#####.#-#####-#.#####----#.#-------#.#-----####.#-#####-#.####--#........#........#--#.##.###.#.###.##.#--#$.#.....@.....#.$#--##.#.#.#####.#.#.##--#....#...#...#....#--#.######.#.######.#--#.................#--###################--#-----------------#-";
