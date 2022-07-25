// uso de crates por la aplicación
use std::error::Error;
use std::io;
use std::process;

// graficar
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::{LineStyle, PointMarker, PointStyle};
use plotlib::view::ContinuousView;

// DBSCAN
use ndarray::{array, ArrayView, Axis};

use petal_clustering::{Dbscan, Fit};
use petal_neighbors::distance::Euclidean;

// parejas de valores
use itertools::Itertools;

// para el test de normalidad de los errores
use statrs::distribution::{ChiSquared, Continuous};
use statrs::statistics::Statistics;

// cañcular la maxima curvatura (codo)
fn calcular_kn_distancia(xx: &Vec<f64>, yy: &Vec<f64>) -> f64 {
    let mut kn_dist = Vec::new();
    let mut sum: f64;
    let mut y: Vec<f64> = yy.clone();
    let mut x: Vec<f64> = xx.clone();

    // ordenar los vectores x,y con base a y
    let mut aswap1: f64;
    let mut aswap2: f64;
    for _j in 0..y.len() - 1 {
        for i in 0..y.len() - 1 {
            if y[i] > y[i + 1] {
                aswap1 = y[i];
                aswap2 = y[i + 1];
                y[i] = aswap2;
                y[i + 1] = aswap1;
                aswap1 = x[i];
                aswap2 = x[i + 1];
                x[i] = aswap2;
                x[i + 1] = aswap1;
            }
        }
    }

    //println!("<<<< y {:?}", y);
    //println!("<<<< x {:?}", x);
    for i in 0..x.len() - 1 {
        sum = f64::powf(x[i] - x[i + 1], 2f64);
        sum += f64::powf(y[i] - y[i + 1], 2f64);
        sum = f64::powf(sum, 0.5);

        kn_dist.push(sum);
    }
    kn_dist.sort_by(cmp_f64);
    // println!(" distancias : {:?}", kn_dist);
    // calcular las pendientes del vector ordenado
    let mut pendientes: Vec<f64> = Vec::new();
    for i in 0..kn_dist.len() - 1 {
        pendientes.push((kn_dist[i + 1] - kn_dist[i]) / ((i as f64 + 1.0) - i as f64));
    }

    // ordenar los vectores pendiente,kn_dist con base a pendiente

    for _j in 0..pendientes.len() - 1 {
        for i in 0..pendientes.len() - 1 {
            if pendientes[i] > pendientes[i + 1] {
                aswap1 = pendientes[i];
                aswap2 = pendientes[i + 1];
                pendientes[i] = aswap2;
                pendientes[i + 1] = aswap1;
                aswap1 = kn_dist[i];
                aswap2 = kn_dist[i + 1];
                kn_dist[i] = aswap2;
                kn_dist[i + 1] = aswap1;
            }
        }
    }
    // buscar la pendiente 1%
    let mut epsi = 0f64;
    let mut m: Vec<f64> = Vec::new();
    let mut d: Vec<f64> = Vec::new();
    for i in 0..pendientes.len() - 1 {
        if pendientes[i] != pendientes[i + 1] {
            m.push(pendientes[i]);
            d.push(kn_dist[i]);
        }
    }
    //println!(" pendientes.... {:?}", pendientes);
    //println!(" distancias.... {:?}", d);
    //println!(" pendientes.... {:?}", m);
    for i in 0..m.len() {
        if m[i] >= 1.0 {
            epsi = d[i];
            break;
        }
    }
    println!(" epsilon para DBSCAN es {}", epsi);
    return epsi;
}

// apoyo a ordenar un vector
use std::cmp::Ordering;

fn cmp_f64(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

// método DAgostino - Pearson como test de normalidad
// Traducción de Matlab en https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46548/versions/2/previews/Codes_data_publish/Codes/Dagostest.m/index.html
// a RUST por enrique buendia lozada
pub fn dago_pear_k2(x: &mut Vec<f64>) -> bool {
    x.sort_by(cmp_f64);

    let alpha: f64 = 0.05;
    let n = x.len();
    let s1: f64 = x.iter().sum(); // suma de los elementos de x
                                  //println!("\ns1 ::: {}\n", s1);
    let s2: f64 = x.iter().map(|a| a.powi(2)).sum::<f64>(); //suma de cada elemento al cuadrado
    let s3: f64 = x.iter().map(|a| a.powi(3)).sum::<f64>(); //suma de cada elemento al cubo
    let s4: f64 = x.iter().map(|a| a.powi(4)).sum::<f64>(); //suma de cada elemento a la 4a

    //println!("\n n: {} s1 {} s2 {} s3 {} s4 {}", n, s1, s2, s3, s4);

    let ss: f64 = s2 - (f64::powf(s1, 2.0) / n as f64);
    let v: f64 = ss / (n as f64 - 1.0);
    let k3: f64 = ((n as f64 * s3) - (3.0 * s1 * s2) + ((2.0 * f64::powf(s1, 3.0)) / n as f64))
        / ((n as f64 - 1.0) * (n as f64 - 2.0));
    let g1: f64 = k3 / f64::powf(f64::powf(v, 3.0), 0.5);
    //println!("\nss {}  v {}  k3 {}   g1{}", ss, v, k3, g1);

    let k4: f64 = ((n as f64 + 1.0)
        * ((n as f64 * s4) - (4.0 * s1 * s3) + (6.0 * (f64::powf(s1, 2.0)) * (s2 / n as f64))
            - ((3.0 * (f64::powf(s1, 4.0))) / (f64::powf(n as f64, 2.0))))
        / ((n as f64 - 1.0) * (n as f64 - 2.0) * (n as f64 - 3.0)))
        - ((3.0 * (f64::powf(ss, 2.0))) / ((n as f64 - 2.0) * (n as f64 - 3.0)));

    let g2: f64 = k4 / f64::powf(v, 2.0);
    let eg1: f64 = ((n as f64 - 2.0) * g1) / f64::powf(n as f64 * (n as f64 - 1.0), 0.5); // sesgo

    //let eg2: f64 = ((n as f64 - 2.0) * (n as f64 - 3.0) * g2) // kurtosis
    //    / ((n as f64 + 1.0) * (n as f64 - 1.0))
    //    + ((3.0 * (n as f64 - 1.0)) / (n as f64 + 1.0));
    //println!("\nk4 {}  g2 {}   eg1 {} ", k4, g2, eg1);

    let a: f64 = eg1
        * f64::powf(
            ((n as f64 + 1.0) * (n as f64 + 3.0)) / (6.0 * (n as f64 - 2.0)),
            0.5,
        );
    let b: f64 = (3.0
        * ((f64::powf(n as f64, 2.0)) + (27.0 * n as f64) - 70.0)
        * ((n as f64 + 1.0) * (n as f64 + 3.0)))
        / ((n as f64 - 2.0) * (n as f64 + 5.0) * (n as f64 + 7.0) * (n as f64 + 9.0));
    let c: f64 = f64::powf(2.0 * (b - 1.0), 0.5) - 1.0;
    let d: f64 = f64::powf(c, 0.5);
    let e: f64 = 1.0 / f64::powf(d.ln(), 0.5);
    let f: f64 = a / f64::powf(2.0 / (c - 1.0), 0.5);
    //println!("a {}  b {}  c {}  d {} e {} f{}", a, b, c, d, e, f);
    let zg1: f64 = e * (f + f64::powf(f64::powf(f, 2.0) + 1.0, 0.5)).ln();
    let g: f64 = (24.0 * n as f64 * (n as f64 - 2.0) * (n as f64 - 3.0))
        / (f64::powf(n as f64 + 1.0, 2.0) * (n as f64 + 3.0) * (n as f64 + 5.0));
    let h: f64 = ((n as f64 - 2.0) * (n as f64 - 3.0) * g2.abs())
        / ((n as f64 + 1.0) * (n as f64 - 1.0) * f64::powf(g, 0.5));
    let j: f64 = ((6.0 * (f64::powf(n as f64, 2.0) - (5.0 * n as f64) + 2.0))
        / ((n as f64 + 7.0) * (n as f64 + 9.0)))
        * f64::powf(
            (6.0 * (n as f64 + 3.0) * (n as f64 + 5.0))
                / (n as f64 * (n as f64 - 2.0) * (n as f64 - 3.0)),
            0.5,
        );
    //println!("\nzg1 {} g {}  h {}  j {}", zg1, g, h, j);
    let k: f64 = 6.0 + ((8.0 / j) * ((2.0 / j) + f64::powf(1.0 + (4.0 / f64::powf(j, 2.0)), 0.5)));
    let l: f64 = (1.0 - (2.0 / k)) / (1.0 + h * f64::powf(2.0 / (k - 4.0), 0.5));
    let zg2: f64 =
        (1.0 - (2.0 / (9.0 * k)) - f64::powf(l, 1. / 3.0)) / f64::powf(2.0 / (9.0 * k), 0.5);
    let k2: f64 = f64::powf(zg1, 2.0) + f64::powf(zg2, 2.0); // D'Agostino-Pearson statistic
                                                             //print!("\nk {} l {} zg2 {} k2: {}", k, l, zg2, k2);
                                                             //println!("\n k2 {}", k2);
    let x2: f64 = k2;
    let df: f64 = 2.0;

    let nn = ChiSquared::new(df).unwrap();
    let prob: f64 = nn.pdf(x2) * 2.0;
    if !x2.is_nan() && !prob.is_nan() {
        println!(" D'Agostino-Pearson normality test: K2 is distributed as Chi-squared with df=2");
        println!(" k2 {}        p {}", x2, prob);

        if prob >= alpha {
            //println!(" Los datos tienen distribución normal");
            return true;
        } else {
            //println!(" Los datos NO tienen distribución normal");
            return false;
        }
    } else {
        println!(
            "  Existe algún tipo de problema y no se puede probar la normalidad en residuales ..."
        );
        return false;
    }
}

pub fn mean(values: &Vec<f64>) -> f64 {
    // promedio de los valores de un vector
    if values.len() == 0 {
        return 0f64;
    }

    return values.iter().sum::<f64>() / (values.len() as f64);
}

pub fn variance(values: &Vec<f64>) -> f64 {
    // varianza de los valores de un vector
    if values.len() == 0 {
        return 0f64;
    }

    let mean = mean(values);
    return values
        .iter()
        .map(|x| f64::powf(x - mean, 2 as f64))
        .sum::<f64>()
        / values.len() as f64;
}

pub fn covariance(x_values: &Vec<f64>, y_values: &Vec<f64>) -> f64 {
    // covarianza de dos vectores del mismo tamaño
    if x_values.len() != y_values.len() {
        panic!("Los vectores x e y deben ser del mismo tamaño...");
    }

    let length: usize = x_values.len();

    if length == 0usize {
        return 0f64;
    }

    let mut covariance: f64 = 0f64;
    let mean_x = mean(x_values);
    let mean_y = mean(y_values);

    for i in 0..length {
        covariance += (x_values[i] - mean_x) * (y_values[i] - mean_y)
    }

    return covariance / length as f64;
}

pub fn r_cuad(yy: &Vec<f64>, esti: &Vec<f64>) {
    // yy es el valor de y actual
    // esti son los valores de las estimaciones
    let med: f64 = mean(&yy);
    let mut sum1: f64 = 0f64;
    let mut sum2: f64 = 0f64;
    for i in 0..yy.len() {
        sum1 += f64::powf(yy[i] - esti[i], 2f64);
        sum2 += f64::powf(yy[i] - med, 2f64);
    }
    // de acuerdo con
    // https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html#:~:text=%C2%AFy)2.-,R%202%20%3D%201%20%E2%88%92%20sum%20squared%20regression%20(SSR)%20total,from%20the%20mean%20all%20squared.
    // https://www.ncl.ac.uk/academic-skills-kit/
    let rdos = 1.0 - sum1 / sum2;
    if !rdos.is_nan() {
        println!("  r_cuad: {}  (0 <= r_cuad <= 1)", rdos);
    } else {
        println!("Hay problema con la iformación...");
    }
}
pub struct LinearRegression {
    pub coefficient: Option<f64>,
    pub intercept: Option<f64>,
}

// regresion lineal simple
impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            coefficient: None,
            intercept: None,
        }
    }
    // crear el modelo
    pub fn fit(&mut self, x_values: &Vec<f64>, y_values: &Vec<f64>) {
        let b1 = covariance(x_values, y_values) / variance(x_values);
        let b0 = mean(y_values) - b1 * mean(x_values);

        self.intercept = Some(b0);
        self.coefficient = Some(b1);
    }
    // calcular una estimacion
    pub fn predict(&self, x: f64) -> f64 {
        if self.coefficient.is_none() || self.intercept.is_none() {
            panic!("fit(..) debe ser llamada primero");
        }

        let b0 = self.intercept.unwrap();
        let b1 = self.coefficient.unwrap();

        return b0 + b1 * x;
    }
    // crear estimaciones de acuerdo con los valores de un vector
    pub fn predict_list(&self, x_values: &Vec<f64>) -> Vec<f64> {
        let mut predictions = Vec::new();

        for i in 0..x_values.len() {
            predictions.push(self.predict(x_values[i]));
        }

        return predictions;
    }
    // RMSE
    pub fn evaluate(&self, x_test: &Vec<f64>, y_test: &Vec<f64>) -> f64 {
        if self.coefficient.is_none() || self.intercept.is_none() {
            panic!("fit(..) debe ser llamado primero...");
        }

        let y_predicted = self.predict_list(x_test);
        return self.root_mean_squared_error(y_test, &y_predicted);
    }
    // pasos de RMSE
    fn root_mean_squared_error(&self, actual: &Vec<f64>, predicted: &Vec<f64>) -> f64 {
        let mut sum_error = 0f64;
        let length = actual.len();

        for i in 0..length {
            sum_error += f64::powf(predicted[i] - actual[i], 2f64);
        }

        let mean_error = sum_error / length as f64;

        return mean_error.sqrt();
    }
}

fn aic_bic(actual: &Vec<f64>, predicted: &Vec<f64>, mut k: f64) {
    //
    // Formulas de AIC y BIC
    // de acuerdo con https://www.sciencedirect.com/science/article/pii/B9780128172162000119
    // Data fitting and regression, Xin-She Yang
    // Introduction to algorithms for data mining and Machine Learning. 2019.
    //
    //

    // k es el numero de variables independientes +1
    k = k + 1.0;
    let mut sum_error = 0f64;
    let mut sum = 0f64;
    let length = actual.len();

    let med = actual.mean();
    for i in 0..length {
        sum_error += f64::powf(predicted[i] - actual[i], 2f64);
        sum += f64::powf(med - actual[i], 2f64);
    }
    let rdos = 1f64 - sum_error / sum; // ecuación 3 de https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8279135/

    println!(
        " R-cuadrada= {} el peor valor -inf y el mejor valor +1",
        rdos
    );
    println!(" RSS= {}", sum_error);

    let aic = 2.0 * k
        + length as f64 * (sum_error / length as f64).ln()
        + (2.0 * k * (k + 1.0)) / (length as f64 - 2.0 - 1.0);
    println!(" AIC= con k={}: {}", k - 1.0, aic);
    let bic = k * (length as f64).ln() + length as f64 * (sum_error / length as f64).ln();
    println!(" BIC= con k={}: {}", k - 1.0, bic);
}

fn estimacion(actual: &Vec<f64>, b0: f64, b1: f64) -> Vec<f64> {
    // estimaciones por separado
    let mut predictions = Vec::new();

    for i in 0..actual.len() {
        predictions.push(b0 + b1 * actual[i]);
    }
    return predictions;
}

fn graficar(x_values: &Vec<f64>, y_values: &Vec<f64>, nombre: String, x: &Vec<f64>, y: &Vec<f64>) {
    // diagrama de dispersion mas línea de regresión, con la ecuación de regresión correspondiente
    let mut model = LinearRegression::new();

    //let intercept = tuple.0;
    //let coefficient = tuple.1;

    model.fit(&x_values, &y_values);

    //let coefficient1 = model.coefficient;
    //let intercept1 = model.intercept;

    let eq = "y=";
    let eq1 = model.coefficient.unwrap().to_string();
    let eq2 = " * x + ";
    let eq3 = model.intercept.unwrap().to_string();
    let eq4 = " ";

    let equat = [eq, &eq1, eq2, &eq3, eq4].join("");

    let y_prediction: Vec<f64> = model.predict_list(&x_values);
    let y_prediction_f64: Vec<f64> = y_prediction.into_iter().map(|x| x as f64).collect();

    let x_values_f64: Vec<f64> = x_values.into_iter().map(|x| *x as f64).collect();
    let y_values_f64: Vec<f64> = y_values.into_iter().map(|x| *x as f64).collect();

    let mut actual: Vec<(f64, f64)> = Vec::new();
    let mut prediction: Vec<(f64, f64)> = Vec::new();
    let mut todos: Vec<(f64, f64)> = Vec::new();

    for i in 0..x_values_f64.len() {
        actual.push((x_values_f64[i], y_values_f64[i]));
        prediction.push((x_values_f64[i], y_prediction_f64[i]));
    }
    for i in 0..x.len() {
        todos.push((x[i], y[i]));
    }

    // límites de los ejes de coordenadas de la gráfica

    let lsx: f64 = Statistics::max(x.iter()) + 1.0;
    let lix: f64 = Statistics::min(x.iter()) - 1.0;
    let lsy: f64 = Statistics::max(y.iter()) + 1.0;
    let liy: f64 = Statistics::min(y.iter()) - 1.0;

    /*let plot_actual = Scatter::from_vec(&actual)
        .style(scatter::Style::new()
            .colour("#35C788"));
    */
    let s0: Plot = Plot::new(todos).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("#808080"),
    );

    let s1: Plot = Plot::new(actual).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("#35c788"),
    );

    /*let plot_prediction = Scatter::from_vec(&prediction)
        .style(scatter::Style::new()
            .marker(Marker::Square)
            .colour("#DD3355"));
    */

    let pred = prediction.clone();
    let s2: Plot = Plot::new(prediction).point_style(
        PointStyle::new() // uses the default marker
            .size(2.0)
            .colour("#dd3355"),
    );

    let s3: Plot = Plot::new(pred).line_style(LineStyle::new().colour("#dd5433"));

    let v = ContinuousView::new()
        .add(s0)
        .add(s1)
        .add(s2)
        .add(s3)
        .x_range(lix, lsx)
        .y_range(liy, lsy)
        .x_label("X")
        .y_label(equat);

    let nom = nombre.clone();
    Page::single(&v).save(nombre).unwrap();
    println!(" Gráfica: {}", nom);
}
fn graficar4(x_values: &Vec<f64>, y_values: &Vec<f64>, nombre: String, x: &Vec<f64>, y: &Vec<f64>) {
    // diagrama de dispersion mas línea de regresión, con la ecuación de regresión correspondiente
    let mut model = LinearRegression::new();

    //let intercept = tuple.0;
    //let coefficient = tuple.1;

    model.fit(&x_values, &y_values);

    //let coefficient1 = model.coefficient;
    //let intercept1 = model.intercept;

    let eq = "y=";
    let eq1 = model.coefficient.unwrap().to_string();
    let eq2 = " * x + ";
    let eq3 = model.intercept.unwrap().to_string();
    let eq4 = " ";

    let equat = [eq, &eq1, eq2, &eq3, eq4].join("");

    let y_prediction: Vec<f64> = model.predict_list(&x_values);
    let y_prediction_f64: Vec<f64> = y_prediction.into_iter().map(|x| x as f64).collect();

    let x_values_f64: Vec<f64> = x_values.into_iter().map(|x| *x as f64).collect();
    let y_values_f64: Vec<f64> = y_values.into_iter().map(|x| *x as f64).collect();

    let mut actual: Vec<(f64, f64)> = Vec::new();
    let mut prediction: Vec<(f64, f64)> = Vec::new();
    let mut todos: Vec<(f64, f64)> = Vec::new();

    for i in 0..x_values_f64.len() {
        actual.push((x_values_f64[i], y_values_f64[i]));
        prediction.push((x_values_f64[i], y_prediction_f64[i]));
    }
    for i in 0..x.len() {
        todos.push((x[i], y[i]));
    }

    // límites de los ejes de coordenadas de la gráfica
    let lsx = Statistics::max(x.iter()) + 1.0; //maximo
    let lix = Statistics::min(x.iter()) - 1.0; //minimo
    let lsy = Statistics::max(y.iter()) + 1.0; //maximo
    let liy = Statistics::min(y.iter()) - 1.0; //minimo

    /*let plot_actual = Scatter::from_vec(&actual)
        .style(scatter::Style::new()
            .colour("#35C788"));
    */
    let s0: Plot = Plot::new(todos).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("#808080"),
    );

    let s1: Plot = Plot::new(actual).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("#35c788"),
    );

    /*let plot_prediction = Scatter::from_vec(&prediction)
        .style(scatter::Style::new()
            .marker(Marker::Square)
            .colour("#DD3355"));
    */

    let pred = prediction.clone();
    let s2: Plot = Plot::new(prediction).point_style(
        PointStyle::new() // uses the default marker
            .size(2.0)
            .colour("#dd3355"),
    );

    let s3: Plot = Plot::new(pred).line_style(LineStyle::new().colour("#dd5433"));

    let v = ContinuousView::new()
        .add(s0)
        .add(s1)
        .add(s2)
        .add(s3)
        .x_range(lix, lsx)
        .y_range(liy, lsy)
        .x_label("X")
        .y_label(equat);

    let nom = nombre.clone();
    let resultado = ["B_", &nom].join("");
    Page::single(&v).save(resultado).unwrap();
    println!(" Gráfica: {}", nom);
}
fn graficar2(x_values: &Vec<f64>, y_values: &Vec<f64>, x: &Vec<f64>, y: &Vec<f64>, nombre: String) {
    // límites de los ejes de coordenadas de la gráfica
    let lsx = Statistics::max(x.iter()) + 1.0; //maximo
    let lix = Statistics::min(x.iter()) - 1.0; //minimo
    let lsy = Statistics::max(y.iter()) + 1.0; //maximo
    let liy = Statistics::min(y.iter()) - 1.0; //minimo

    let mut actual: Vec<(f64, f64)> = Vec::new();
    let mut fuera: Vec<(f64, f64)> = Vec::new();

    for i in 0..x.len() {
        actual.push((x[i], y[i]));
    }
    for i in 0..x_values.len() {
        fuera.push((x_values[i], y_values[i]));
    }

    let s1: Plot = Plot::new(actual).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("GREEN"),
    );

    let s2: Plot = Plot::new(fuera).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("MAGENTA"),
    );

    let v = ContinuousView::new()
        .add(s1)
        .add(s2)
        .x_range(lix, lsx)
        .y_range(liy, lsy)
        .x_label("x")
        .y_label("y");

    let nom = nombre.clone();
    Page::single(&v).save(nombre).unwrap();
    println!("  Gráfica: {} ", nom);
}
fn graficar5(x_values: &Vec<f64>, y_values: &Vec<f64>, x: &Vec<f64>, y: &Vec<f64>, nombre: String) {
    // límites de los ejes de coordenadas de la gráfica
    let lsx = Statistics::max(x.iter()) + 1.0; //maximo
    let lix = Statistics::min(x.iter()) - 1.0; //minimo
    let lsy = Statistics::max(y.iter()) + 1.0; //maximo
    let liy = Statistics::min(y.iter()) - 1.0; //minimo

    let mut actual: Vec<(f64, f64)> = Vec::new();
    let mut fuera: Vec<(f64, f64)> = Vec::new();

    for i in 0..x.len() {
        actual.push((x[i], y[i]));
    }
    for i in 0..x_values.len() {
        fuera.push((x_values[i], y_values[i]));
    }

    let s1: Plot = Plot::new(actual).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("GREEN"),
    );

    let s2: Plot = Plot::new(fuera).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("MAGENTA"),
    );

    let v = ContinuousView::new()
        .add(s1)
        .add(s2)
        .x_range(lix, lsx)
        .y_range(liy, lsy)
        .x_label("x")
        .y_label("y");

    let nom = nombre.clone();
    let resultado = ["B_", &nom].join("");
    Page::single(&v).save(resultado).unwrap();
    println!("  Gráfica: {} ", nom);
}
pub fn residuos(act: &Vec<f64>, est: &Vec<f64>) -> Vec<f64> {
    // residuales= y_actaul - y_estimada
    let mut residuales: Vec<f64> = Vec::new();

    for i in 0..act.len() {
        residuales.push(act[i] - est[i]);
    }
    if dago_pear_k2(&mut residuales) {
        println!(" Los residuales SI tienen distribución normal.")
    }

    return residuales;
}

// separar las mediciones por grupos de 6 lpm
pub fn bandas(x: &Vec<f64>, y: &Vec<f64>, nombre: String, id_v: &Vec<String>) {
    if x.len() > 7 {
        let mut model = LinearRegression::new();
        let mut escala: f64;

        // vectores de subgrupo
        let mut yy: Vec<f64> = Vec::new();
        let mut xx: Vec<f64> = Vec::new();
        let mut id: Vec<String> = Vec::new();
        // límites de x e y
        let lsy = Statistics::max(y.iter()); //maximo
        let liy = Statistics::min(y.iter()); //minimo

        let fin = lsy;
        let mut contador: u32 = 0;
        escala = liy;
        //println!(">>> bandas...");
        while escala + 6.0 <= fin {
            for j in 0..x.len() {
                if y[j] >= escala && y[j] < escala + 6.0 {
                    // 6.0 es el ancho de la subgrupo
                    xx.push(x[j]);
                    yy.push(y[j]);
                    id.push(id_v[j].to_string());
                }
            }
            if xx.len() >= 2 {
                println!(" ___");
                println!(">>> y tomadas para la banda {:?}", yy);
                println!(">>> x tomada para la banda {:?}", xx);
                println!(">>> id correspondiente {:?}", id);
                let maxy = Statistics::max(yy.iter());
                let miny = Statistics::min(yy.iter());
                println!("  y  max={} min={}", maxy, miny);

                let residuales = yy.clone();

                // graficar subgrupo de valores
                let aa: &str = "banda_";
                let b: String = (contador + 1).to_string();
                let bb: &str = "_";
                let cc: &str = ".svg";
                let result2: String = [aa, &nombre, bb, &b, cc].join("");
                //let result2 = [aa, &b, cc].join("");
                graficar(&xx, &yy, result2, &x, &y);
                // crear modelo lineal ---------------
                model.fit(&xx, &yy);
                if !model.coefficient.unwrap().is_nan() && !model.intercept.unwrap().is_nan() {
                    println!(">>> Coeficiente : {0}", model.coefficient.unwrap());
                    println!(">>> Intercepción: {0}", model.intercept.unwrap());
                    let b0: f64 = model.intercept.unwrap();
                    let b1: f64 = model.coefficient.unwrap();

                    let esti: Vec<f64> = estimacion(&xx, b0, b1);
                    aic_bic(&xx, &esti, 1.0);
                    println!(" Residuales: {:?}", residuos(&residuales, &esti));
                    println!(">>> RMSE   : {0}", model.evaluate(&xx, &yy));
                    r_cuad(&yy, &esti);
                    println!(
                        ">>> y= {}*x + {} \n",
                        model.coefficient.unwrap(),
                        model.intercept.unwrap()
                    );
                } else {
                    println!(
                        "  ... Hay algún tipo de problema y no se puede construir el modelo ..."
                    );
                }
            }
            // el +6.0 es debido a que la recta de regresion pasara por el promedio
            // y el objetivo es que no exista mucha variación de lo estimado con la realidad
            // de ahi que las estimaciones solo variaran en aproximadamente 3 valores
            escala = escala + 6.0;
            //liy = liy + escala;
            //lsy = lsy + escala;
            xx.clear();
            yy.clear();
            id.clear();
            contador += 1;
        }
    }
}
// separar las mediciones por grupos de 6 lpm
pub fn bandas4(x: &Vec<f64>, y: &Vec<f64>, nombre: String, id_v: &Vec<String>) {
    if x.len() > 7 {
        let mut model = LinearRegression::new();
        let mut escala: f64;

        // vectores de subgrupo
        let mut yy: Vec<f64> = Vec::new();
        let mut xx: Vec<f64> = Vec::new();
        let mut id: Vec<String> = Vec::new();
        // límites de x e y

        let lsy = Statistics::max(y.iter()); //maximo
        let liy = Statistics::min(y.iter()); //minimo
        let fin = lsy;
        let mut contador: u32 = 0;
        escala = liy;
        //println!(">>> bandas...");
        while escala + 6.0 <= fin {
            for j in 0..x.len() {
                if y[j] >= escala && y[j] < escala + 6.0 {
                    // 6.0 es el ancho de la subgrupo
                    xx.push(x[j]);
                    yy.push(y[j]);
                    id.push(id_v[j].to_string());
                }
            }
            if xx.len() >= 2 {
                println!(" ___");
                println!("B>>> y tomadas para la banda {:?}", yy);
                println!("B>>> x tomada para la banda {:?}", xx);
                println!("B>>> id correspondiente {:?}", id);
                let maxy = Statistics::max(yy.iter());
                let miny = Statistics::min(yy.iter());
                println!("  y  max={} min={}", maxy, miny);

                let residuales = yy.clone();

                // graficar subgrupo de valores
                let aa: &str = "B_banda_";
                let b: String = (contador + 1).to_string();
                let bb: &str = "_";
                let cc: &str = ".svg";
                let result2: String = [aa, &nombre, bb, &b, cc].join("");
                //let result2 = [aa, &b, cc].join("");
                graficar(&xx, &yy, result2, &x, &y);
                // crear modelo lineal ---------------
                model.fit(&xx, &yy);
                if !model.coefficient.unwrap().is_nan() && !model.intercept.unwrap().is_nan() {
                    println!("B>>> Coeficiente : {0}", model.coefficient.unwrap());
                    println!("B>>> Intercepción: {0}", model.intercept.unwrap());
                    let b0: f64 = model.intercept.unwrap();
                    let b1: f64 = model.coefficient.unwrap();

                    let esti: Vec<f64> = estimacion(&xx, b0, b1);
                    aic_bic(&xx, &esti, 1.0);
                    println!(" Residuales: {:?}", residuos(&residuales, &esti));
                    println!("B>>> RMSE   : {0}", model.evaluate(&xx, &yy));
                    r_cuad(&yy, &esti);
                    println!(
                        "B>>> y= {}*x + {} \n",
                        model.coefficient.unwrap(),
                        model.intercept.unwrap()
                    );
                } else {
                    println!(
                        "  ... Hay algún tipo de problema y no se puede construir el modelo ..."
                    );
                }
            }
            // el +6.0 es debido a que la recta de regresion pasara por el promedio
            // y el objetivo es que no exista mucha variación de lo estimado con la realidad
            // de ahi que las estimaciones solo variaran en aproximadamente 3 valores
            escala = escala + 6.0;
            //liy = liy + escala;
            //lsy = lsy + escala;
            xx.clear();
            yy.clear();
            id.clear();
            contador += 1;
        }
    }
}
// algoritmo DBSCAN
fn f_dbscan(x: &Vec<f64>, y: &Vec<f64>, id_v: &Vec<String>) {
    // Create a couple of iterators
    let x_iter = x.iter();
    let y_iter = y.iter();

    // Interleave x_iter with y_iter and group into tuples of size 2 using itertools
    let mut v = Vec::new();
    for (a, b) in x_iter.interleave(y_iter).tuples() {
        v.push((*a, *b)); // If I don't de-reference with *, I get Vec<(&f64, &f64)>
    }

    //println!("v= {:?} ",v); *****************

    let mut points = array![
        [1.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.3],
        [8.0, 7.0],
        [8.0, 8.0],
        [25.0, 80.0]
    ];
    //points.row_mut(0)[0] = 6.66;
    let zeros = ArrayView::from(&[0.; 2]).into_shape((1, 2)).unwrap();

    // apendizar pares necesarios
    let tam: i32;

    tam = x.len() as i32 - 5;

    for _ in 1..tam {
        let result = points.append(Axis(0), zeros);
        match result {
            Ok(_result) => {}
            Err(e) => {
                println!(" {:?}", e);
            }
        }
    }
    // copiar todo x e y a points

    for i in 0..x.len() {
        points.row_mut(i)[0] = x[i];
        points.row_mut(i)[1] = y[i];
    }
    //println!(">>> Datos que se pasaran a DBSCAN ...\n{:?}", points);
    println!("\n\n___\n>>> Inicia DBSCAN ...\n");

    // vectores para los clusters encontrados
    let mut yy: Vec<f64> = Vec::new(); // variable dependiente
    let mut xx: Vec<f64> = Vec::new(); // variable independiente
    let mut id: Vec<String> = Vec::new(); // identificadores

    // radio de vecindad = 2.0=eps
    // El número mínimo de puntos necesarios para formar una región densa = 3=min_samples
    let min_s = 3i32;
    let epsi = calcular_kn_distancia(&x, &y); // estimar epsilon por curvatura maxima

    let clustering =
        Dbscan::new(epsi, min_s.try_into().unwrap(), Euclidean::default()).fit(&points);
    let mut model = LinearRegression::new();
    println!(
        "\n ======= Clusters encontrados {:?}  =======",
        clustering.0.len()
    ); // two clusters found
    for i in 0..clustering.0.len() {
        println!("\n -------> Puntos en cluster {} {:?}", i, clustering.0[&i]); // the first three points in Cluster 0
                                                                                //println!("Puntos en cluster 1 {:?}", clustering.0[&1]); // [8.0, 7.0] and [8.0, 8.0] in Cluster 1
                                                                                // formar los vectores para el ajuste
        for m in clustering.0[&i].iter() {
            //println!("** {}",m);

            xx.push(x[*m]);
            yy.push(y[*m]);
            id.push(id_v[*m].to_string());
        }
        let residuales = yy.clone();
        println!(" == y -> {:?}", yy);
        println!(" == x -> {:?}", xx);
        println!(" == id-> {:?}", id);
        let maxy = Statistics::max(yy.iter());
        let miny = Statistics::min(yy.iter());
        println!("  y  max={} min={}", maxy, miny);
        // crear modelo lineal ---------------

        //let intercept = tuple.0;
        //let coefficient = tuple.1;

        model.fit(&xx, &yy);

        //let coefficient1 = model.coefficient;
        //let intercept1 = model.intercept;
        if !(model.coefficient.unwrap()).is_nan()
            && !(model.intercept.unwrap()).is_nan()
            && !(model.evaluate(&xx, &yy)).is_nan()
            && !((model.coefficient.unwrap()) == 0.0)
        {
            let b0: f64 = model.intercept.unwrap();
            let b1: f64 = model.coefficient.unwrap();
            println!("\n\n Coeficiente : {0}", b1);
            println!(" Intercepción: {0}", b0);

            let esti = estimacion(&xx, b0, b1);
            aic_bic(&xx, &esti, 1.0);
            println!(" Residuales: {:?}", residuos(&residuales, &esti));
            println!(" RMSE   : {0}", model.evaluate(&xx, &yy));
            r_cuad(&yy, &esti);
            println!(
                "    y= {}*x + {} \n",
                model.coefficient.unwrap(),
                model.intercept.unwrap()
            );

            // construir nombre de archivo grafico correspondiente al cluster
            let a = "DBSCAN_Cluster";
            let b = (i).to_string();
            let c = ".svg";
            let result = [a, &b, c].join("");

            let cxx = xx.clone();
            let cyy = yy.clone();
            graficar(&xx, &yy, result, &x, &y);
            // crear los modelos con el menor error
            bandas(&cxx, &cyy, b, &id);
            // -----------------------------------
        } else {
            println!("  ... Hay algún tipo de problema y no se puede construir el modelo ...");
        }
        xx.clear();
        yy.clear();
        id.clear();
    }
    println!(
        "\n___\n DBSCAN: Puntos fuera de clusters {:?}",
        clustering.1
    ); // [25.0, 80.0] doesn't belong to any cluster
    xx.clear();
    yy.clear();
    id.clear();
    for m in clustering.1.iter() {
        xx.push(x[*m]);
        yy.push(y[*m]);
        id.push(id_v[*m].to_string());
    }
    println!(" y: {:?}", yy);
    println!(" x: {:?}", xx);
    println!(" id:{:?}", id);
    let maxy = Statistics::max(yy.iter());
    let miny = Statistics::min(yy.iter());
    println!("  y  max={} min={}", maxy, miny);
    let result = "puntos_fuera_por_DBSCAN.svg";
    graficar2(&xx, &yy, &x, &y, result.to_string());
}

// algoritmo DBSCAN
fn f_dbscan2(x: &Vec<f64>, y: &Vec<f64>, id_v: &Vec<String>) {
    // Create a couple of iterators
    let x_iter = x.iter();
    let y_iter = y.iter();

    // Interleave x_iter with y_iter and group into tuples of size 2 using itertools
    let mut v = Vec::new();
    for (a, b) in x_iter.interleave(y_iter).tuples() {
        v.push((*a, *b)); // If I don't de-reference with *, I get Vec<(&f64, &f64)>
    }

    //println!("v= {:?} ",v); *****************

    let mut points = array![
        [1.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.3],
        [8.0, 7.0],
        [8.0, 8.0],
        [25.0, 80.0]
    ];
    //points.row_mut(0)[0] = 6.66;
    let zeros = ArrayView::from(&[0.; 2]).into_shape((1, 2)).unwrap();

    // apendizar pares necesarios
    let tam: i32;

    tam = x.len() as i32 - 5;

    for _ in 1..tam {
        let result = points.append(Axis(0), zeros);
        match result {
            Ok(_result) => {}
            Err(e) => {
                println!(" {:?}", e);
            }
        }
    }
    // copiar todo x e y a points

    for i in 0..x.len() {
        points.row_mut(i)[0] = x[i];
        points.row_mut(i)[1] = y[i];
    }
    //println!("\n\n>>> B inicia DBSCAN con épsilon = 3...\n{:?}", points);
    println!("\n\n___\n>>> B inicia DBSCAN con épsilon = 3"); // de acuerdo con https://www.asep.org/asep/asep/Robergs2.pdf
    println!(">>> B inicia DBSCAN min_samples = 4..."); // aquí se usó https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf ester
                                                        // usar el numero de veces que se toma la FC a cada persona en el experimento
                                                        // vectores para los clusters encontrados
    let mut yy: Vec<f64> = Vec::new(); // variable dependiente
    let mut xx: Vec<f64> = Vec::new(); // variable independiente
    let mut id: Vec<String> = Vec::new(); // identificadores

    // radio de vecindad = 3.0=eps
    // El número mínimo de puntos necesarios para formar una región densa = 4=min_samples
    let min_s = 4i32; // minimo de puntos en cluster de acuerdo a https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
    let epsi = 3f64; //  https://www.asep.org/asep/asep/Robergs2.pdf

    let clustering =
        Dbscan::new(epsi, min_s.try_into().unwrap(), Euclidean::default()).fit(&points);
    let mut model = LinearRegression::new();
    println!(
        "\n ======= B Clusters encontrados {:?}  =======",
        clustering.0.len()
    ); // two clusters found
    for i in 0..clustering.0.len() {
        println!(
            "\n -------> B Puntos en cluster {} {:?}",
            i, clustering.0[&i]
        ); // the first three points in Cluster 0
           //println!("Puntos en cluster 1 {:?}", clustering.0[&1]); // [8.0, 7.0] and [8.0, 8.0] in Cluster 1
           // formar los vectores para el ajuste
        for m in clustering.0[&i].iter() {
            //println!("** {}",m);

            xx.push(x[*m]);
            yy.push(y[*m]);
            id.push(id_v[*m].to_string());
        }
        let residuales = yy.clone();
        println!(" == y -> {:?}", yy);
        println!(" == x -> {:?}", xx);
        println!(" == id-> {:?}", id);
        let maxy = Statistics::max(yy.iter());
        let miny = Statistics::min(yy.iter());
        println!("  y  max={} min={}", maxy, miny);
        // crear modelo lineal ---------------

        //let intercept = tuple.0;
        //let coefficient = tuple.1;

        model.fit(&xx, &yy);

        //let coefficient1 = model.coefficient;
        //let intercept1 = model.intercept;
        if !(model.coefficient.unwrap()).is_nan()
            && !(model.intercept.unwrap()).is_nan()
            && !(model.evaluate(&xx, &yy)).is_nan()
            && !((model.coefficient.unwrap()) == 0.0)
        {
            let b0: f64 = model.intercept.unwrap();
            let b1: f64 = model.coefficient.unwrap();
            println!("\n\n Coeficiente : {0}", b1);
            println!(" Intercepción: {0}", b0);

            let esti = estimacion(&xx, b0, b1);
            aic_bic(&xx, &esti, 1.0);
            println!(" Residuales: {:?}", residuos(&residuales, &esti));
            println!(" RMSE   : {0}", model.evaluate(&xx, &yy));
            r_cuad(&yy, &esti);
            println!(
                "    y= {}*x + {} \n",
                model.coefficient.unwrap(),
                model.intercept.unwrap()
            );

            // construir nombre de archivo grafico correspondiente al cluster
            let a = "B_DBSCAN_Cluster";
            let b = (i).to_string();
            let c = ".svg";
            let result = [a, &b, c].join("");

            let cxx = xx.clone();
            let cyy = yy.clone();
            graficar4(&xx, &yy, result, &x, &y);
            // crear los modelos con el menor error
            bandas4(&cxx, &cyy, b, &id);
            // -----------------------------------
        } else {
            println!("  ... Hay algún tipo de problema y no se puede construir el modelo ...");
        }
        xx.clear();
        yy.clear();
        id.clear();
    }
    println!(
        "\n___\n B DBSCAN: Puntos fuera de clusters {:?}",
        clustering.1
    ); // [25.0, 80.0] doesn't belong to any cluster
    xx.clear();
    yy.clear();
    id.clear();
    for m in clustering.1.iter() {
        xx.push(x[*m]);
        yy.push(y[*m]);
        id.push(id_v[*m].to_string());
    }
    println!(" y: {:?}", yy);
    println!(" x: {:?}", xx);
    println!(" id:{:?}", id);
    let maxy = Statistics::max(yy.iter());
    let miny = Statistics::min(yy.iter());
    println!("  y  max={} min={}", maxy, miny);
    let result = "B_puntos_fuera_por_DBSCAN.svg";
    graficar5(&xx, &yy, &x, &y, result.to_string());
}

// organizar de la actividad
pub fn analisis() -> Result<(), Box<dyn Error>> {
    let mut model = LinearRegression::new();

    // -----
    // Crear el lector CSV e iterar sobre cada registro.
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut y_values: Vec<f64> = Vec::new();
    let mut x_values: Vec<f64> = Vec::new();
    let mut id_values = Vec::new();

    for result in rdr.records() {
        // leer lineas correspondientes del archivo csv.
        //
        let record = result?;

        // convertir a flotante lo correspondiente
        y_values.push(record[0].parse::<f64>().unwrap());
        x_values.push(record[1].parse::<f64>().unwrap());
        id_values.push(record[2].to_string()); // identificadores
    }
    // println!("{:?}", id_values); // mostrar el vector construido
    // -----

    model.fit(&x_values, &y_values);

    println!(
        "\n Enrique R.P. Buendia Lozada 2022. Benemérita Universidad Autónoma de Puebla, México."
    );
    // Regresión lineal por clusters definidos por el algorítmo DBSCAN
    println!(" DBSCAN de Machine Learning de Inteligencia Artificial");
    println!("\n\n\n Ecuación con todos los datos originales del archivo de entrada...");
    if !model.coefficient.unwrap().is_nan() && !model.intercept.unwrap().is_nan() {
        println!("\n\n Coeficiente : {0}", model.coefficient.unwrap());
        println!(" Intercepción: {0}", model.intercept.unwrap());
        println!(" RMSE   : {0}", model.evaluate(&x_values, &y_values));
        println!(
            "    y= {}*x + {} \n",
            model.coefficient.unwrap(),
            model.intercept.unwrap()
        );
    } else {
        println!("  ... Hay algún tipo de problema y no se puede construir el modelo ...\n");
    }
    let y_prediction: Vec<f64> = model.predict_list(&x_values);
    let y_prediction_f64: Vec<f64> = y_prediction.into_iter().map(|x| x as f64).collect();
    r_cuad(&y_values, &y_prediction_f64);
    let x_values_f64: Vec<f64> = x_values.into_iter().map(|x| x as f64).collect();
    let y_values_f64: Vec<f64> = y_values.into_iter().map(|x| x as f64).collect();

    let mut actual: Vec<(f64, f64)> = Vec::new();
    let mut prediction: Vec<(f64, f64)> = Vec::new();

    for i in 0..x_values_f64.len() {
        actual.push((x_values_f64[i], y_values_f64[i]));
        prediction.push((x_values_f64[i], y_prediction_f64[i]));
    }

    // límites de los ejes de coordenadas de la gráfica
    let lsx = Statistics::max(x_values_f64.iter()) + 1.0; //maximo
    let lix = Statistics::min(x_values_f64.iter()) - 1.0; //minimo
    let lsy = Statistics::max(y_values_f64.iter()) + 1.0; //maximo
    let liy = Statistics::min(y_values_f64.iter()) - 1.0; //minimo

    let s1: Plot = Plot::new(actual).point_style(
        PointStyle::new()
            .size(1.0)
            .marker(PointMarker::Square) // definir los marcadores como cuadros
            .colour("#35c788"),
    );

    let pred = prediction.clone();
    let s2: Plot = Plot::new(prediction).point_style(
        PointStyle::new() // uses the default marker
            .size(2.0)
            .colour("#dd3355"),
    );

    let s3: Plot = Plot::new(pred).line_style(LineStyle::new().colour("#dd5433"));

    let v = ContinuousView::new()
        .add(s1)
        .add(s2)
        .add(s3)
        .x_range(lix, lsx)
        .y_range(liy, lsy)
        .x_label("X")
        .y_label("Y");

    let nombre = "scatter.svg";
    let nom = nombre.clone();
    Page::single(&v).save(nombre).unwrap();
    println!(" Gráfica: {}", nom);

    let cxx = x_values_f64.clone();
    let cyy = y_values_f64.clone();
    // crear los modelos con el menor error sin quitar patrones no consecutivos de mediciones
    // con los datos originales crear bandas
    println!("\n ___... bandas sin tomar DBSCAN ....___");
    bandas(&cxx, &cyy, "orig_".to_string(), &id_values);
    println!("\n ___... bandas tomando en cuenta DBSCAN ....___");
    // DBSCAN para identificar patrones no consecutivos de mediciones
    f_dbscan(&x_values_f64, &y_values_f64, &id_values); // epsilon via maxima curvatura
    f_dbscan2(&x_values_f64, &y_values_f64, &id_values); // epsilon = 3

    Ok(())
}

use glob::glob;
use std::env;
fn remueve_f() {
    // borra todas las gráficas con extensión svg en el directorio de trabajo
    // para que cada análisis esté solo lo que corresponda
    let dir = env::current_dir().unwrap();
    // println!(" {:?}", dir);
    let cad = dir.into_os_string().into_string().unwrap();
    let result = str::replace(&cad, "\\", "/");
    let rr = result.as_str();
    println!(
        " Carpeta de trabajo actual ... {:?}\n  borrando *.svg...",
        rr
    );
    let resu = [rr, "/*.svg"].join("");

    for path in glob(&resu).unwrap() {
        match path {
            Ok(path) => {
                //println!("Removiendo archivo: {:?}", path.display());
                let _msg = std::fs::remove_file(path);
                //println!("Error {:?}", msg);
            }
            Err(e) => {
                println!("Error {:?}", e);
            }
        };
    }
}

pub fn main() {
    // Autor: Enrique Ricardo Pablo Buendia Lozada
    // 2022
    //

    // remover información anterior, para no crear confusión en los resultados
    remueve_f();
    // iniciar análisis de la información donde (x) puede ser la edad y (y) la frecuencia cardiaca
    if let Err(err) = analisis() {
        println!("error en la actividad de implementación: {}", err);
        process::exit(1);
    }
}
