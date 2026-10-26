/// Repli de bord : le pixel extérieur miroir le pixel intérieur, bord répété.
pub fn reflect_index(index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }
    let period = length * 2;
    let mut index = index % period;
    if index < 0 {
        index += period;
    }
    if index >= length {
        index = period - index - 1;
    }
    index
}

pub fn clamp_index(index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }
    if index < 0 {
        0
    } else if index >= length {
        length - 1
    } else {
        index
    }
}
