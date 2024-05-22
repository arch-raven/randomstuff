use rand::{thread_rng, Rng};
use std::collections::HashSet;

const THRESH: usize = 1000;

fn flip_coin() -> bool {
    let mut rng = thread_rng();
    rng.gen_bool(0.5)
}

fn test_flip_coin(n_trials: u32) {
    let x = (0..n_trials).map(|_| flip_coin() as u32).sum::<u32>() as f32;
    let p = x / (n_trials as f32);
    println!("[test_flip_coin] p = {}, true_p = 0.5", p);
}

fn n_consecutive_heads(n: u32) -> bool {
    (0..n).all(|_| flip_coin())
}

fn test_n_consecutive_heads(n_trials: u32, n_heads: u32) {
    let x = (0..n_trials)
        .map(|_| n_consecutive_heads(n_heads) as u32)
        .sum::<u32>() as f32;
    let p = x / (n_trials as f32);
    println!(
        "[test_n_consecutive_heads] p = {}, true_p = {}",
        p,
        1.0 / 2_i32.pow(n_heads) as f32
    );
}

fn cvm<T: std::hash::Hash + std::cmp::Eq>(tokens: impl Iterator<Item = T>) -> usize {
    let mut round: u32 = 0;
    let mut vocab = HashSet::new();
    for tok in tokens {
        if vocab.len() == THRESH {
            vocab.retain(|_| flip_coin());
            round += 1;
        }
        vocab.remove(&tok);
        if n_consecutive_heads(round) {
            vocab.insert(tok);
        }
    }

    let result = vocab.len() * 2_usize.pow(round);
    // println!( "round: {round} vocab_length: {} Result: {result}",vocab.len());
    result
}

fn test_cvm() {
    let stream_length: usize = 100000;
    let mut rng = rand::thread_rng();
    let tokens: Vec<i32> = (0..stream_length).map(|_| rng.gen_range(0..3600)).collect();
    let true_unk_tokens = HashSet::<i32>::from_iter(tokens.clone()).len();
    let pred_unk_tokens = cvm(tokens.iter());
    println!("[test_cvm] thresh: {THRESH} pred: {pred_unk_tokens} | true: {true_unk_tokens}");
}

fn main() {
    let n_trials: u32 = 100000;
    test_flip_coin(n_trials);
    test_n_consecutive_heads(n_trials, 6);
    test_cvm();
}
