// Exact canonical codec permutation generators, extracted for independent controls.
pub fn generate_weight_perm_int4() -> [usize; 1024] {
    let mut perm = [0usize; 1024];
    let mut idx = 0;

    for i in 0..32 {
        let col = i / 4;
        let mut perm1 = [0usize; 8];
        let mut p1_idx = 0;

        for block in 0..2 {
            for &row in &[
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1[p1_idx] = 16 * row + col + 8 * block;
                p1_idx += 1;
            }
        }

        for j in 0..4 {
            for &p in &perm1 {
                perm[idx] = p + 256 * j;
                idx += 1;
            }
        }
    }

    // Apply INT4 interleaving: [0, 2, 4, 6, 1, 3, 5, 7]
    let interleave = [0, 2, 4, 6, 1, 3, 5, 7];
    let mut result = [0usize; 1024];
    for group in 0..(1024 / 8) {
        for (dest, &src) in interleave.iter().enumerate() {
            result[group * 8 + dest] = perm[group * 8 + src];
        }
    }

    result
}
pub fn generate_weight_perm_int8() -> [usize; 1024] {
    let mut perm = [0usize; 1024];
    let mut idx = 0;

    for i in 0..32 {
        let col = i / 4;
        let mut perm1 = [0usize; 8];
        let mut p1_idx = 0;

        for block in 0..2 {
            for &row in &[
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1[p1_idx] = 16 * row + col + 8 * block;
                p1_idx += 1;
            }
        }

        for j in 0..4 {
            for &p in &perm1 {
                perm[idx] = p + 256 * j;
                idx += 1;
            }
        }
    }

    // Apply INT8 interleaving: [0, 2, 1, 3] (groups of 4, not 8)
    let interleave = [0, 2, 1, 3];
    let mut result = [0usize; 1024];
    for group in 0..(1024 / 4) {
        for (dest, &src) in interleave.iter().enumerate() {
            result[group * 4 + dest] = perm[group * 4 + src];
        }
    }

    result
}
pub fn generate_scale_perms() -> ([usize; 64], [usize; 32]) {
    // scale_perm: [0,8,16,24,32,40,48,56, 1,9,17,25,... 7,15,23,...,63]
    let mut scale_perm = [0usize; 64];
    for i in 0..8 {
        for j in 0..8 {
            scale_perm[i * 8 + j] = i + 8 * j;
        }
    }

    // scale_perm_single: for channelwise
    let offsets = [0, 1, 8, 9, 16, 17, 24, 25];
    let mut scale_perm_single = [0usize; 32];
    for i in 0..4 {
        for (j, &off) in offsets.iter().enumerate() {
            scale_perm_single[i * 8 + j] = 2 * i + off;
        }
    }

    (scale_perm, scale_perm_single)
}
