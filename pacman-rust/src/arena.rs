use crate::config::*;
use crossterm::{
    execute,
    terminal::{Clear, ClearType},
};

pub struct Arena {
    pub static_board: Vec<Vec<Icons>>,
    pub board: Vec<Vec<Icons>>,
    pub pallete_count: u32,
}

impl Arena {
    pub fn new() -> Self {
        let board = vec![vec![Icons::Empty; WIDTH]; HEIGHT];
        let static_board = vec![vec![Icons::Empty; WIDTH]; HEIGHT];
        let mut arena = Self {
            board,
            static_board,
            pallete_count: 0,
        };
        arena.read_board();

        for icon in arena.board.iter().flatten() {
            match icon {
                Icons::Pellet | Icons::PowerPellet => arena.pallete_count += 1,
                _ => (),
            }
        }
        arena
    }

    pub fn read_board(&mut self) {
        let grid: Vec<char> = GRID_LAYOUT.chars().collect();
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                self.static_board[i][j] = Icons::from_char(grid[i * WIDTH + j]);
            }
        }
        self.board = self.static_board.clone();
    }

    pub fn init_icon_position(&mut self, icon: Icons) -> Coordinates {
        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                if self.static_board[i][j] == icon {
                    self.static_board[i][j] = Icons::Empty;
                    return Coordinates::new(j as i32, i as i32);
                }
            }
        }
        Coordinates::new(-1, -1)
    }

    pub fn display(&self) {
        execute!(std::io::stdout(), Clear(ClearType::All)).unwrap();
        for row in self.board.iter() {
            for icon in row.iter() {
                print!("{} ", icon.to_char());
            }
            println!();
        }
    }

    pub fn static_display(&self) {
        for row in self.static_board.iter() {
            for icon in row.iter() {
                print!("{}", icon.to_char());
            }
            println!();
        }
    }
}
