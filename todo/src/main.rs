use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;

use plotters::prelude::*;
use csv::Reader;

use crate::full_palette::GREY;

fn read_csv(file_path: &str) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;
    let mut data = Vec::new();

    for result in reader.records() {
        let record = result?;
        let y: f64 = record.get(0).unwrap().parse()?;
        let x: f64 = record.get(1).unwrap().parse()?;
        data.push((x, y));
    }

    Ok(data)
}

/// Función para crear la gráfica con ecuaciones y puntos del CSV
fn create_equation_graph(
    equations: &[(f64, f64, String)],
    csv_data: &[(f64, f64)],
) -> Result<(), Box<dyn Error>> {
    // Calcular los rangos mínimo y máximo para x e y basados en los datos CSV
    let csv_x_values: Vec<f64> = csv_data.iter().map(|(x, _)| *x).collect();
    let csv_y_values: Vec<f64> = csv_data.iter().map(|(_, y)| *y).collect();

    let x_min = csv_x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let x_max = csv_x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let y_min = csv_y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let y_max = csv_y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;

    // Crear una nueva imagen para la gráfica
    let root = BitMapBackend::new("todo.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Configurar la gráfica
    let mut chart = ChartBuilder::on(&root)
        .caption("Ecuaciones de Regresión Lineal y Puntos CSV", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    // Dibujar la malla
    chart.configure_mesh().draw()?;

    // Dibujar cada ecuación en la gráfica
    for (i, (coef, intercept, _eq_str)) in equations.iter().enumerate() {
        // Definir colores diferentes para cada línea
        let color = match i % 5 {
            0 => &RED,
            1 => &BLUE,
            2 => &GREEN,
            3 => &BLACK,
            4 => &MAGENTA,
            _ => &GREY,
        };

        // Crear la función de la línea de regresión
        let f = |x: f64| coef * x + intercept;

        // Calcular los valores x para la línea de regresión dentro del rango de los datos CSV
        let x_vals = (x_min as i64..x_max as i64 + 1)
            .map(|x| x as f64)
            .collect::<Vec<_>>();

        // Dibujar la línea de regresión
        chart.draw_series(LineSeries::new(
            x_vals.iter().map(|x| (*x, f(*x))),
            color,
        ))?;

        // Agregar la ecuación a la leyenda
        //chart.configure_series_labels()
        //    .label_font(("sans-serif", 10))
        //    .label(format!("Línea {}: {}", i + 1, eq_str))
        //    .draw()?;
        // Mostrar la ecuación de la regresión
        let equation = format!("y = {:.3}x + {:.3}", coef, intercept);
        //println!("Ecuación de la regresión: {}", equation);

        // Calcular el punto medio de la línea de regresión
        let x_mid = (x_min + x_max) / 2.0;
        let y_mid = coef * x_mid + intercept;

        // Agregar la ecuación como texto sobre la línea de regresión
        chart.draw_series(std::iter::once(Text::new(equation, (x_mid, y_mid), ("Arial", 16).into_font())))?;

    }

    // Dibujar los puntos del CSV
    //chart.draw_series(csv_data.iter().map(|&(x, y)| Circle::new((x, y), 5, &RED)))?;
    // **********************************************************************************
    // Abrir el archivo salida.txt
    let file = File::open("salida.txt")?;
    let reader = BufReader::new(file);

    // Variables para almacenar los bloques de datos
    let mut clusters: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut x_data: Vec<f64> = Vec::new();
    let mut y_data: Vec<f64> = Vec::new();

    // Bandera para identificar si estamos dentro de un bloque relevante
    let mut in_cluster_block = false;

    // Leer el archivo línea por línea
    for line in reader.lines() {
        let line = line?;

        if line.starts_with(" -------> B Puntos en cluster") {
            // Inicio de un nuevo bloque de puntos
            in_cluster_block = true;
            x_data.clear();
            y_data.clear();
        } else if in_cluster_block && line.starts_with(" == y -> ") {
            // Extraer los valores de y
            let values: Vec<&str> = line
                .trim_start_matches(" == y -> ")
                .trim_matches(|c| c == '[' || c == ']') // Eliminar corchetes
                .split(", ")
                .collect();
            for v in values {
                match v.trim().parse::<f64>() {
                    Ok(num) => y_data.push(num),
                    Err(_) => eprintln!("Error al parsear valor 'y': {}", v),
                }
            }
        } else if in_cluster_block && line.starts_with(" == x -> ") {
            // Extraer los valores de x
            let values: Vec<&str> = line
                .trim_start_matches(" == x -> ")
                .trim_matches(|c| c == '[' || c == ']') // Eliminar corchetes
                .split(", ")
                .collect();
            for v in values {
                match v.trim().parse::<f64>() {
                    Ok(num) => x_data.push(num),
                    Err(_) => eprintln!("Error al parsear valor 'x': {}", v),
                }
            }
        } else if in_cluster_block && line.is_empty() {
            // Fin del bloque, guardar los datos
            if !x_data.is_empty() && !y_data.is_empty() {
                clusters.push((x_data.clone(), y_data.clone()));
            }
            in_cluster_block = false; // Reiniciar para el siguiente bloque
        }
    }

    // Graficar todos los bloques con colores diferentes
    // plot_clusters(&clusters)?;
    // Definir una paleta de colores
    let colors = [RED, BLUE, GREEN, BLACK, MAGENTA, CYAN, YELLOW,
                    RGBColor(255, 0, 0),       // Rojo puro
                    RGBColor(0, 255, 0),       // Verde puro
                    RGBColor(0, 0, 255),       // Azul puro
                    RGBColor(255, 255, 0),     // Amarillo
                    RGBColor(128, 0, 128),     // Púrpura
                    RGBColor(0, 128, 128),     // Turquesa
                    RGBColor(255, 165, 0),     // Naranja
                    RGBColor(75, 0, 130),      // Índigo
                    RGBColor(255, 192, 203),   // Rosa
                    RGBColor(135, 206, 235),   // Celeste
                    RGBColor(128, 128, 0),     // Oliva
                    RGBColor(165, 42, 42),     // Marrón
    ];

    // Dibujar cada cluster con un color diferente
    for (i, (x, y)) in clusters.iter().enumerate() {
        let color = colors[i % colors.len()]; // Asignar un color único al cluster

        chart.draw_series(PointSeries::of_element(
            x.iter().zip(y.iter()).map(|(&x_val, &y_val)| (x_val, y_val)),
            5,
            &color,
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
        ))?;
    }

    // Guardar la gráfica
    root.present()?;
    Ok(())
}



fn create_graph(data: &[(f64, f64)]) -> Result<(), Box<dyn Error>> {
    // Calcular los rangos mínimo y máximo para x e y
    let x_values: Vec<f64> = data.iter().map(|&(x, _)| x).collect();
    let y_values: Vec<f64> = data.iter().map(|&(_, y)| y).collect();
    
    let x_min = x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let x_max = x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    let y_min = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let y_max = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;

    let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gráfica de Datos", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(data.iter().map(|&(x, y)| Circle::new((x, y), 5, &RED)))?;

    root.present()?;
    Ok(())
}


fn read_equations_from_file(file_path: &str) -> Result<Vec<(f64, f64, String)>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut equations = Vec::new();

    for line in reader.lines() {
        let line = line?;
        println!("Procesando línea: {}", line); // Mensaje de depuración

        if line.starts_with(">>> y=") || line.starts_with("y=") {
            // Dividir la línea en partes
            let parts: Vec<&str> = line.split_whitespace().collect();
            let parts = if parts.first() == Some(&">>>") {
                parts.into_iter().skip(1).collect::<Vec<_>>() // Eliminar el primer elemento
            } else {
                parts
            };

            // Imprimir las partes procesadas
            println!("{:?}", parts);

            // Inicializar variables
            let mut num1: f64 = 0.0;
            let mut intercept: f64 = 0.0;

            // Encontrar la parte que contiene el '*'
            if let Some(coef_part) = parts.iter().find(|&&s| s.contains('*')) {
                // Separar la cadena en dos partes usando '*'
                if let Some((num_part, _)) = coef_part.split_once('*') {
                    // Eliminar espacios en blanco alrededor de la parte numérica
                    let num_part = num_part.trim();

                    // Convertir la parte numérica a f64
                    match num_part.parse::<f64>() {
                        Ok(parsed_num) => {
                            num1 = parsed_num;
                            println!("Número extraído: {}", num1);
                        }
                        Err(e) => {
                            println!("Error al convertir el número: {:?}", e);
                            continue; // Saltar a la siguiente iteración del bucle
                        }
                    }
                } else {
                    println!("No se encontró '*' en la cadena.");
                    continue; // Saltar a la siguiente iteración del bucle
                }
            } else {
                println!("No se encontró '*' en ninguna parte del vector.");
                continue; // Saltar a la siguiente iteración del bucle
            }

            // Extraer el intercepto
            if parts.len() > 3 {
                match parts[3].parse::<f64>() {
                    Ok(parsed_intercept) => {
                        intercept = parsed_intercept;
                    }
                    Err(e) => {
                        println!("Error al convertir el intercepto: {:?}", e);
                        continue; // Saltar a la siguiente iteración del bucle
                    }
                }
            } else {
                println!("No se encontró intercepto en la cadena.");
                continue; // Saltar a la siguiente iteración del bucle
            }

            // Agregar la ecuación al vector
            equations.push((num1, intercept, line.clone()));
            println!("Ecuación encontrada: {}", line); // Mensaje de depuración
        }
    }

    println!("Ecuaciones encontradas: {:?}", equations); // Mensaje de depuración
    Ok(equations)
}

fn main() -> Result<(), Box<dyn Error>> {
    let equations = read_equations_from_file("salida.txt")?;
    println!("Ecuaciones leídas del archivo: {:?}", equations);


     // Leer los datos del archivo CSV
     let data = read_csv("fc.csv")?;

     // Crear la gráfica
     create_graph(&data)?;

     // Crear la gráfica con las ecuaciones y los puntos del CSV
     create_equation_graph(&equations, &data)?;
 
     println!("Gráfica generada con éxito.");
     
    Ok(())
}