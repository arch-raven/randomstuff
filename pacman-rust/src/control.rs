use crate::config::Coordinates;

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
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    match input.trim() {
        "w" => UserInput::Up,
        "s" => UserInput::Down,
        "a" => UserInput::Left,
        "d" => UserInput::Right,
        "q" => UserInput::Quit,
        _ => UserInput::Invalid,
    }
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
