# dip_model2_adaptive.py
# ===========================================================
# Versión extendida con pesos dinámicos para L_cheap
# Mantiene lógica original del entrenamiento DIP
# ===========================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# -----------------------------------------------------------
# Total Variation
# -----------------------------------------------------------
def total_variation_loss(x):
    dy = x[:, 1:, :, :] - x[:, :-1, :, :]
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    return tf.reduce_mean(tf.abs(dx)) + tf.reduce_mean(tf.abs(dy))

# -----------------------------------------------------------
# Simple UNet-like generator
# -----------------------------------------------------------
def make_generator(input_shape=(32,32,16), out_channels=1, base_filters=64):
    inp = layers.Input(shape=input_shape)
    x = inp

    # Bloque 1
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2)(x)

    # Bloque 2
    x = layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters*2, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=2)(x)

    # Salida
    out = layers.Conv2D(out_channels, 3, padding='same')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

# -----------------------------------------------------------
# DIP con pesos dinámicos
# -----------------------------------------------------------
def run_dip_partial_boundary(
        cheap, boundary, mask,
        noise_shape=(32,32,16),
        iters=5000,
        lr=1e-3,
        weight_cheap_inside=0.1,
        weight_tv=1e-4,
        print_every=200,
        patience=800
    ):

    n, m = cheap.shape
    cheap = cheap.astype(np.float32)
    boundary = boundary.astype(np.float32)
    mask = mask.astype(np.float32)

    # Normalización
    global_max = max(np.max(np.abs(cheap)), np.max(np.abs(boundary)), 1.0)
    cheap_n = cheap / global_max
    boundary_n = boundary / global_max
    mask_n = mask

    # Generador
    gen = make_generator(input_shape=noise_shape, out_channels=1, base_filters=64)
    optimizer = optimizers.Adam(learning_rate=lr)

    # Ruido fijo
    z = np.random.normal(size=(1, noise_shape[0], noise_shape[1], noise_shape[2])).astype(np.float32)

    # Historial
    history = {'loss_total': [], 'loss_boundary': [], 'loss_cheap': [], 'loss_tv': [], 'w_cheap': []}
    best_loss = np.inf
    best_pred = None
    wait = 0

    for it in range(1, iters+1):
        # Jitter opcional
        z_jitter = z + 0.03 * np.random.normal(size=z.shape).astype(np.float32)

        # Peso dinámico del cheap
        w_cheap = weight_cheap_inside * (1 - it / iters)
        w_bnd = 1.0  # Fijo

        with tf.GradientTape() as tape:
            out = gen(z_jitter, training=True)  # (1,H,W,1)
            out_resized = tf.image.resize(out, (n, m), method='bilinear')

            # Pérdida frontera
            diff_boundary = out_resized[0,:,:,0] - boundary_n
            loss_boundary = tf.reduce_sum(tf.square(diff_boundary * mask_n)) / (tf.reduce_sum(mask_n) + 1e-8)

            # Pérdida interior vs cheap
            diff_inside = out_resized[0,:,:,0] - cheap_n
            loss_cheap = tf.reduce_sum(tf.square(diff_inside * (1.0 - mask_n))) / (tf.reduce_sum(1.0 - mask_n) + 1e-8)

            # TV
            loss_tv = total_variation_loss(out_resized)

            # Pérdida total con pesos dinámicos
            loss = w_bnd * loss_boundary + w_cheap * loss_cheap + weight_tv * loss_tv

        grads = tape.gradient(loss, gen.trainable_variables)
        optimizer.apply_gradients(zip(grads, gen.trainable_variables))

        # Historial
        history['loss_total'].append(float(loss.numpy()))
        history['loss_boundary'].append(float(loss_boundary.numpy()))
        history['loss_cheap'].append(float(loss_cheap.numpy()))
        history['loss_tv'].append(float(loss_tv.numpy()))
        history['w_cheap'].append(float(w_cheap))

        # Best model según frontera
        if history['loss_boundary'][-1] < best_loss:
            best_loss = history['loss_boundary'][-1]
            best_pred = out_resized.numpy()[0,:,:,0].copy()
            wait = 0
        else:
            wait += 1

        # Logs
        if it % print_every == 0 or it == 1:
            print(
                f"it {it:05d} | loss={loss.numpy():.6e} "
                f"| bnd={loss_boundary.numpy():.6e} "
                f"cheap={loss_cheap.numpy():.6e} "
                f"tv={loss_tv.numpy():.6e} "
                f"| w_cheap={w_cheap:.3e}"
            )

        if wait > patience:
            print("Early stopping por paciencia.")
            break

    # Reconstrucción final
    out_final = best_pred if best_pred is not None else out_resized.numpy()[0,:,:,0]
    out_final = out_final * global_max

    # Restaurar valores conocidos
    out_final[mask == 1] = boundary[mask == 1]

    return out_final, history
