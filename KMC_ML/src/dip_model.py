import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks

# -------------------------
# Generador (encoder-decoder simple)
# -------------------------
def make_generator(input_shape=(32, 32, 16), out_channels=1, base_filters=64):
    """
    Generador encoder-decoder simple. input_shape = (H_noise, W_noise, C_noise).
    Devuelve un modelo Keras que mapea ruido->mapa (resolución libre; se resizea fuera).
    """
    inp = layers.Input(shape=input_shape)
    x = inp

    # Encoder
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')(x)

    # Decoder / Upsample
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(x)

    out = layers.Conv2D(out_channels, 1, padding='same', activation='linear')(x)
    return Model(inputs=inp, outputs=out)


# -------------------------
# Pérdida TV
# -------------------------
def total_variation_loss(x):
    # x: tensor shape (batch,H,W,1)
    dh = tf.abs(x[:, 1:, :, :] - x[:, :-1, :, :])
    dw = tf.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return tf.reduce_mean(dh) + tf.reduce_mean(dw)


# -------------------------
# Run DIP con pesos adaptativos (mejorado)
# -------------------------
def run_dip_adaptive(
    cheap, boundary, mask,
    noise_shape=(32, 32, 16),
    iters=5000,
    lr=1e-3,
    weight_boundary=1.0,
    weight_cheap_initial=0.1,
    weight_tv=1e-4,
    weight_cheap_schedule=None,   # función (it, total_iters) -> peso
    print_every=200,
    patience=800,
    seed=None,
    device="/GPU:0"
):
    """
    Entrenamiento DIP con pesos adaptativos.
    - weight_cheap_schedule: opcional, función que recibe (it, iters) y devuelve peso.
      Si None, se usa decaimiento lineal: w = weight_cheap_initial*(1 - it/iters).
    - seed: reproducibilidad opcional
    - device: dispositivo, por ejemplo '/GPU:0' o '/CPU:0'
    """

    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    n, m = cheap.shape
    cheap = cheap.astype(np.float32)
    boundary = boundary.astype(np.float32)
    mask = mask.astype(np.float32)

    # normalización (constante scala)
    global_max = float(max(np.max(np.abs(cheap)), np.max(np.abs(boundary)), 1.0))
    cheap_n = cheap / global_max
    boundary_n = boundary / global_max
    mask_n = mask.astype(np.float32)

    # convertir constantes a TF para eficiencia (una sola vez)
    cheap_tf = tf.constant(cheap_n, dtype=tf.float32)
    boundary_tf = tf.constant(boundary_n, dtype=tf.float32)
    mask_tf = tf.constant(mask_n, dtype=tf.float32)
    inv_mask_tf = tf.constant(1.0 - mask_n, dtype=tf.float32)

    # crear generador
    gen = make_generator(input_shape=noise_shape, out_channels=1, base_filters=64)
    optimizer = optimizers.Adam(learning_rate=lr)

    # ruido inicial fijo (batch size 1)
    z = np.random.normal(size=(1, noise_shape[0], noise_shape[1], noise_shape[2])).astype(np.float32)

    history = {'loss_total': [], 'loss_boundary': [], 'loss_cheap': [], 'loss_tv': [], 'w_cheap': []}
    best_loss = np.inf
    best_pred = None
    wait = 0

    # Device context (opcional)
    with tf.device(device):
        for it in range(1, iters + 1):
            # calcular peso para 'cheap'
            if callable(weight_cheap_schedule):
                weight_cheap = float(weight_cheap_schedule(it, iters))
            else:
                weight_cheap = float(weight_cheap_initial * (1.0 - float(it) / float(iters)))

            # jitter (pequeño ruido adicional en la entrada)
            z_jitter = z + 0.03 * np.random.normal(size=z.shape).astype(np.float32)

            with tf.GradientTape() as tape:
                out = gen(z_jitter, training=True)  # (1,Hgen,Wgen,1)
                out_resized = tf.image.resize(out, (n, m), method='bilinear')  # (1,n,m,1)

                pred = out_resized[0, :, :, 0]  # (n,m)

                # pérdidas (usar tensores preconstruidos)
                diff_b = pred - boundary_tf
                loss_boundary = tf.reduce_sum(tf.square(diff_b * mask_tf)) / (tf.reduce_sum(mask_tf) + 1e-8)

                diff_in = pred - cheap_tf
                loss_cheap = tf.reduce_sum(tf.square(diff_in * inv_mask_tf)) / (tf.reduce_sum(inv_mask_tf) + 1e-8)

                loss_tv = total_variation_loss(out_resized)

                loss = (weight_boundary * loss_boundary +
                        weight_cheap * loss_cheap +
                        weight_tv * loss_tv)

            grads = tape.gradient(loss, gen.trainable_variables)
            optimizer.apply_gradients(zip(grads, gen.trainable_variables))

            # guardar historial
            history['loss_total'].append(float(loss.numpy()))
            history['loss_boundary'].append(float(loss_boundary.numpy()))
            history['loss_cheap'].append(float(loss_cheap.numpy()))
            history['loss_tv'].append(float(loss_tv.numpy()))
            history['w_cheap'].append(weight_cheap)

            # guardar mejor segun loss_boundary
            if history['loss_boundary'][-1] < best_loss:
                best_loss = history['loss_boundary'][-1]
                best_pred = pred.numpy().copy()
                wait = 0
            else:
                wait += 1

            if it % print_every == 0 or it == 1:
                print(f"it {it:05d} | total={loss.numpy():.3e} "
                      f"| bnd={loss_boundary.numpy():.3e} "
                      f"| cheap={loss_cheap.numpy():.3e} "
                      f"| tv={loss_tv.numpy():.3e} "
                      f"| w_cheap={weight_cheap:.4f}")

            if wait > patience:
                print("Early stopping triggered.")
                break

    # reconstrucción final y post-procesado
    out_final = best_pred if best_pred is not None else pred.numpy()
    out_final = out_final * global_max
    out_final[mask == 1] = boundary[mask == 1]

    return out_final, history, gen
