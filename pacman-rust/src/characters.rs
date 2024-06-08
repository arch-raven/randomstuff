use crate::arena::Arena;
use crate::config::{self, Coordinates, Icons, DIRECTIONS, OVERLORD_TEAM_SIZE};
use crate::control::UserInput;
use rand::seq::SliceRandom;
use rand::thread_rng;

struct Ghost {
    icon: Icons,
    position: Coordinates,
    direction_id: usize,
    direction_order: [i32; 4],
}

impl Default for Ghost {
    fn default() -> Self {
        Self {
            icon: Icons::Ghost,
            position: Coordinates::new(0, 0),
            direction_id: 0,
            direction_order: [-1, 0, 1, 2],
        }
    }
}

impl Ghost {
    pub fn new(arena: &mut Arena) -> Self {
        let position = arena.init_icon_position(Icons::Ghost);
        Self {
            icon: Icons::Ghost,
            position,
            direction_id: 0,
            direction_order: [-1, 0, 1, 2],
        }
    }

    pub fn move_character(&mut self, arena: &mut Arena) {
        let (x, y) = self.position.to_tuple();
        self.direction_order.shuffle(&mut thread_rng());
        for rand_dir in self.direction_order {
            let d_id = (self.direction_id as i32 + rand_dir).rem_euclid(4) as usize;
            let new_position = self.position + DIRECTIONS[d_id];
            let (nx, ny) = new_position.to_tuple();
            if arena.board[ny][nx] != Icons::Wall {
                arena.board[y][x] = arena.static_board[ny][nx];
                arena.board[ny][nx] = self.icon;
                self.position = new_position;
                self.direction_id = d_id;
                break;
            }
        }
    }
}

pub struct Pacman {
    icon: Icons,
    position: Coordinates,
    direction: Coordinates,
    pub status: bool,
    invincible_timer: u32,
    pub score: u32,
}

impl Pacman {
    pub fn new(arena: &mut Arena) -> Self {
        let position = arena.init_icon_position(Icons::Pacman);
        Self {
            icon: Icons::Pacman,
            position,
            direction: Coordinates::new(0, 0),
            status: true,
            invincible_timer: 0,
            score: 0,
        }
    }

    pub fn move_character(&mut self, arena: &mut Arena, user: UserInput) {
        self.direction = match user {
            UserInput::Invalid => self.direction,
            _ => user.to_coordinates(),
        };
        let (x, y) = self.position.to_tuple();
        let new_position = self.position + self.direction;
        let (nx, ny) = new_position.to_tuple();
        if arena.board[ny][nx] == Icons::Wall {
            return;
        }
        arena.board[y][x] = Icons::Empty;
        self.eat_pellet(arena, new_position);
        arena.board[ny][nx] = self.icon;
        self.position = new_position;
    }

    fn eat_pellet(&mut self, arena: &mut Arena, position: Coordinates) {
        let (x, y) = position.to_tuple();
        self.score += match arena.static_board[y][x] {
            Icons::Pellet => config::PALLETE_POINTS,
            Icons::PowerPellet => config::POWER_PALLETE_POINTS,
            _ => 0,
        };
        arena.static_board[y][x] = Icons::Empty;
        arena.pallete_count -= 1;
    }
}

pub struct Overlord {
    pub frightened_mode: bool,
    ghosts: [Ghost; OVERLORD_TEAM_SIZE],
}

impl Overlord {
    pub fn new(arena: &mut Arena) -> Self {
        let mut ghosts: [Ghost; OVERLORD_TEAM_SIZE] = Default::default();
        for i in 0..OVERLORD_TEAM_SIZE {
            ghosts[i] = Ghost::new(arena);
        }
        Self {
            frightened_mode: false,
            ghosts,
        }
    }

    pub fn work(&mut self, arena: &mut Arena, pacman: &mut Pacman) {
        self.update_ghost_mode(pacman);
        self.check_for_headon_collision(pacman);
        for ghost in self.ghosts.iter_mut() {
            ghost.move_character(arena);
            if ghost.position == pacman.position && !self.frightened_mode {
                pacman.status = false;
            }
        }
    }

    fn update_ghost_mode(&mut self, pacman: &mut Pacman) {
        if pacman.invincible_timer > 0 {
            self.frightened_mode = true;
            pacman.invincible_timer -= 1;
        } else {
            self.frightened_mode = false;
        }
    }

    fn check_for_headon_collision(&self, pacman: &mut Pacman) {
        for ghost in self.ghosts.iter() {
            if ghost.position == pacman.position
                && DIRECTIONS[ghost.direction_id] == pacman.direction
                && !self.frightened_mode
            {
                pacman.status = false;
            }
        }
    }
}
