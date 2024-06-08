pub mod arena;
pub mod characters;
pub mod config;
pub mod control;

fn main() {
    let mut arena = arena::Arena::new();
    let mut pacman = characters::Pacman::new(&mut arena);
    let mut overlord = characters::Overlord::new(&mut arena);

    control::greet_player();
    while arena.pallete_count > 0 && pacman.status {
        arena.display();
        let user_input = control::get_user_input();
        if user_input == control::UserInput::Quit {
            break;
        }
        pacman.move_character(&mut arena, user_input);
        overlord.work(&mut arena, &mut pacman);
    }

    control::exit_player(pacman.score);
}
