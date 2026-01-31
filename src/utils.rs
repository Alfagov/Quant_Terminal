pub fn unzip3(v: Vec<(f64, f64, f64)>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut a = Vec::with_capacity(v.len());
    let mut b = Vec::with_capacity(v.len());
    let mut c = Vec::with_capacity(v.len());
    for (x, y, z) in v {
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

pub fn unzip8(
    v: Vec<(f64, f64, f64, f64, f64, f64, f64, f64)>,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let mut a = Vec::with_capacity(v.len());
    let mut b = Vec::with_capacity(v.len());
    let mut c = Vec::with_capacity(v.len());
    let mut d = Vec::with_capacity(v.len());
    let mut e = Vec::with_capacity(v.len());
    let mut f = Vec::with_capacity(v.len());
    let mut g = Vec::with_capacity(v.len());
    let mut h = Vec::with_capacity(v.len());
    for (v1, v2, v3, v4, v5, v6, v7, v8) in v {
        a.push(v1);
        b.push(v2);
        c.push(v3);
        d.push(v4);
        e.push(v5);
        f.push(v6);
        g.push(v7);
        h.push(v8);
    }
    (a, b, c, d, e, f, g, h)
}