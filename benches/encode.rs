use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_encode_flat(c: &mut Criterion) {
    let game = fow_rl::fuzzing::new_default_game();
    c.bench_function("encode_flat", |b| {
        b.iter(|| {
            let encoded = fow_rl::fuzzing::encode_flat(black_box(&game));
            black_box(encoded);
        })
    });
}

fn bench_apply_and_encode(c: &mut Criterion) {
    let base = fow_rl::fuzzing::new_default_game();
    let moves = fow_rl::fuzzing::legal_actions(&base);
    if moves.is_empty() {
        return;
    }
    let mv = moves[0];

    c.bench_function("apply_and_encode", |b| {
        b.iter(|| {
            let mut game = base.clone();
            fow_rl::fuzzing::apply(&mut game, mv);
            let encoded = fow_rl::fuzzing::encode_flat(black_box(&game));
            black_box(encoded);
        })
    });
}

criterion_group!(benches, bench_encode_flat, bench_apply_and_encode);
criterion_main!(benches);
