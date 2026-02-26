"""
dip_model2.py
-------------
Implementación del entrenamiento Deep Image Prior (DIP) para reconstrucción
de matrices Monte Carlo a partir de una matriz barata y valores conocidos.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


# ----------------------------------------------------------------------
# Arquitectura del generador
# ----------------------------------------------------------------------
def make_generator(input_shape=(32, 32, 16), out_channels=1, base_filters=64):
    """
    Construye una red convolucional encoder–decoder (generador) usada
    para el esquema Deep Image Prior.

    Parameters
    ----------
    input_shape : tuple
        Tamaño del ruido de entrada (H, W, C).
    out_channels : int
        Número de canales de salida (1 por defecto).
    base_filters : int
        Número de filtros base en la primera capa.

    Returns
    -------
    keras.Model
        Modelo generador.
    """
    inp = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(base_filters * 4, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(base_filters * 4, 3, padding="same", activation="relu")(x)

    # Decoder
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(base_filters * 2, 3, padding="same", activation="relu")(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(x)

    out = layers.Conv2D(out_channels, 1, padding="same", activation="linear")(x)
    return Model(inputs=inp, outputs=out)


# ----------------------------------------------------------------------
# Regularización: Total Variation Loss
# ----------------------------------------------------------------------
def total_variation_loss(x):
    """
    Calcula la pérdida de variación total (TV) para suavizar la imagen.

    Parameters
    ----------
    x : tf.Tensor
        Tensor de salida del generador (batch, H, W, C).

    Returns
    -------
    tf.Tensor
        Escalar con la magnitud total de la variación.
    """
    dh = tf.abs(x[:, 1:, :, :] - x[:, :-1, :, :])
    dw = tf.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return tf.reduce_mean(dh) + tf.reduce_mean(dw)


# ----------------------------------------------------------------------
# Entrenamiento DIP clásico (sin pesos adaptativos)
# ----------------------------------------------------------------------
def run_dip_partial_boundary(
    cheap,
    boundary,
    mask,
    noise_shape=(32, 32, 16),
    iters=5000,
    lr=1e-3,
    weight_cheap_inside=0.1,
    weight_tv=1e-4,
    print_every=200,
    patience=800,
):
    """
    Entrena la red DIP usando la información de frontera parcial.

    Parameters
    ----------
    cheap : np.ndarray
        Matriz barata (completa).
    boundary : np.ndarray
        Matriz con valores conocidos en frontera, resto 0.
    mask : np.ndarray
        Máscara binaria (1 donde se conocen valores).
    noise_shape : tuple
        Forma del tensor de ruido de entrada.
    iters : int
        Número máximo de iteraciones.
    lr : float
        Tasa de aprendizaje.
    weight_cheap_inside : float
        Peso de la pérdida asociada a la matriz barata en el interior.
    weight_tv : float
        Peso de la regularización de variación total.
    print_every : int
        Frecuencia de impresión de progreso.
    patience : int
        Iteraciones sin mejora antes de early stopping.

    Returns
    -------
    tuple
        (out_final, history)
        - out_final: matriz reconstruida final.
        - history: diccionario con evolución de pérdidas.
    """

    n, m = cheap.shape
    cheap = cheap.astype(np.float32)
    boundary = boundary.astype(np.float32)
    mask = mask.astype(np.float32)

    # Normalización global
    global_max = max(np.max(np.abs(cheap)), np.max(np.abs(boundary)), 1.0)
    cheap_n = cheap / global_max
    boundary_n = boundary / global_max
    mask_n = mask

    # Crear modelo y optimizador
    gen = make_generator(input_shape=noise_shape, out_channels=1, base_filters=64)
    optimizer = optimizers.Adam(learning_rate=lr)

    # Ruido fijo de entrada
    z = np.random.normal(size=(1, *noise_shape)).astype(np.float32)

    # Inicializar historial
    history = {"loss_total": [], "loss_boundary": [], "loss_cheap": [], "loss_tv": []}
    best_loss = np.inf
    best_pred = None
    wait = 0

    # Loop de entrenamiento
    for it in range(1, iters + 1):
        z_jitter = z + 0.03 * np.random.normal(size=z.shape).astype(np.float32)

        with tf.GradientTape() as tape:
            out = gen(z_jitter, training=True)  # (1,H,W,1)
            out_resized = tf.image.resize(out, (n, m), method="bilinear")
            pred = out_resized[0, :, :, 0]

            # Pérdida de frontera
            diff_boundary = pred - boundary_n
            loss_boundary = tf.reduce_sum(tf.square(diff_boundary * mask_n)) / (
                tf.reduce_sum(mask_n) + 1e-8
            )

            # Pérdida interior (cheap)
            diff_inside = pred - cheap_n
            loss_cheap = tf.reduce_sum(tf.square(diff_inside * (1.0 - mask_n))) / (
                tf.reduce_sum(1.0 - mask_n) + 1e-8
            )

            # Regularización TV
            loss_tv = total_variation_loss(out_resized)

            # Total
            loss = loss_boundary + weight_cheap_inside * loss_cheap + weight_tv * loss_tv

        grads = tape.gradient(loss, gen.trainable_variables)
        optimizer.apply_gradients(zip(grads, gen.trainable_variables))

        # Guardar historial
        history["loss_total"].append(float(loss.numpy()))
        history["loss_boundary"].append(float(loss_boundary.numpy()))
        history["loss_cheap"].append(float(loss_cheap.numpy()))
        history["loss_tv"].append(float(loss_tv.numpy()))

        # Mejor modelo
        if history["loss_boundary"][-1] < best_loss:
            best_loss = history["loss_boundary"][-1]
            best_pred = pred.numpy().copy()
            wait = 0
        else:
            wait += 1

        # Logging
        if it % print_every == 0 or it == 1:
            print(
                f"it {it:05d} | loss={loss.numpy():.6e} | "
                f"bnd={loss_boundary.numpy():.6e} | "
                f"cheap={loss_cheap.numpy():.6e} | tv={loss_tv.numpy():.6e}"
            )

        # Early stopping
        if wait > patience:
            print("Early stopping por paciencia.")
            break

    # Salida final
    out_final = best_pred if best_pred is not None else pred.numpy()
    out_final = out_final * global_max
    out_final[mask == 1] = boundary[mask == 1]

    return out_final, history
