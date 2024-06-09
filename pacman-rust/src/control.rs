use crate::config::Coordinates;
use crossterm::event::{read, Event, KeyCode};

#[derive(PartialEq, Eq)]
pub enum UserInput {
    Up,
    Down,
    Left,
    Right,
    Quit,
    Invalid,
}

pub fn get_user_input() -> UserInput {
    let mut input = UserInput::Invalid;
    if let Event::Key(key_event) = read().unwrap() {
        input = match key_event.code {
            KeyCode::Up => UserInput::Up,
            KeyCode::Down => UserInput::Down,
            KeyCode::Left => UserInput::Left,
            KeyCode::Right => UserInput::Right,
            KeyCode::Char('q') => UserInput::Quit,
            _ => UserInput::Invalid,
        };
    }
    input
}

impl UserInput {
    pub fn to_coordinates(&self) -> Coordinates {
        match self {
            UserInput::Left => Coordinates::new(-1, 0),
            UserInput::Up => Coordinates::new(0, -1),
            UserInput::Down => Coordinates::new(0, 1),
            UserInput::Right => Coordinates::new(1, 0),
            _ => Coordinates::new(0, 0),
        }
    }
}

pub fn greet_player() {
    println!("Welcome to Pacman!");
}

pub fn exit_player(score: u32) {
    println!("Game Over, Goodbye!");
    println!("Your score is: {}", score)
}
