from matplotlib import pyplot as plt
import numpy as np

def srgb_to_linearrgb(c):
    if   c < 0:       return 0
    elif c < 0.04045: return c/12.92
    else:             return ((c+0.055)/1.055)**2.4


def hex_to_rgb(h,alpha=1):
    # source: https://blender.stackexchange.com/questions/153094/blender-2-8-python-how-to-set-material-color-using-hex-value-instead-of-rgb
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return tuple([srgb_to_linearrgb(c/0xff) for c in (r,g,b)] + [alpha])

def pprint(array, precision=4):
    with np.printoptions(
        precision=precision,
        suppress=True,
    ):
        print(array)

def get_range(eps=0.1, n=100, symmetric=False, eps_negative=None):
    if eps_negative is None:
        eps_negative = eps
    taus = np.linspace((-1 if symmetric else 0)-eps_negative, 1+eps, n)
    taus = np.insert(taus, np.searchsorted(taus, [0, 1]), [0, 1])
    return taus

def get_grid(min, max, n=100):
    import jax.numpy as jnp
    if isinstance(n, int):
        n = [n for _ in range(len(min))]
    axis_values = (jnp.linspace(min[i], max[i], n[i]) for i in range(len(min)))
    grid_values = jnp.meshgrid(*axis_values, indexing='ij')
    return jnp.squeeze(jnp.stack(grid_values, axis=-1))

def get_bins(n, *values):
    all_values = np.hstack(values)
    return np.linspace(all_values.min(), all_values.max(), n)

def lighten_color(color, amount=0.5):
    """
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_df(df, **kwargs):
    conds = [df[col] == val for col, val in kwargs.items()]
    cond = conds[0]
    for i in range(1, len(conds)):
        cond = cond & conds[i]
    return df[cond]

def get_df_single(df, **kwargs):
   df_subset = get_df(df, **kwargs)
   if len(df_subset) != 1:
      raise ValueError(f"Expected unique result for {kwargs=}. Got {df_subset}")
   return df_subset.iloc[0]

def bold(text):
    text = text.replace('_', ' ')
    text = text.replace(' ', '}$ $\\bf{')
    return r"$\bf{" + text + r"}$"

from matplotlib import colors
class NonSymmetricNormalize(colors.Normalize):
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        if self.vmin is None or self.vmax is None:
            self.autoscale_None(result)
        # Convert at least to float, without losing precision.
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        if vmin == vmax:
            result.fill(0)  # Or should it be all masked?  Or 0.5?
        elif vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                        mask=mask)
            # ma division is very slow; we can take a shortcut
            resdat = result.data
            resdat[resdat < 0] /= np.abs(vmin) * 2
            resdat[resdat > 0] /= np.abs(vmax) * 2
            resdat += .5
            # resdat /= (vmax - vmin)
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result
cmap = 'RdBu_r'

def plot_sdf_surface(ax, inputs, sdf_values, n_positive_levels=10, n_negative_levels=10, surface_width=2, with_0_level=True, hide_ticks=True, **kwargs):
    contour = ax.contourf(
        inputs[..., 0], inputs[..., 1], NonSymmetricNormalize()(sdf_values), cmap='RdBu_r',
        levels=np.hstack([np.linspace(0, .5, n_negative_levels), np.linspace(0.5, 1., n_positive_levels+1)[1:]])
    )

    # make sure there are no white gaps between level sets
    for c in contour.collections:
        c.set_edgecolor("face")

    if with_0_level:
        ax.contour(inputs[..., 0], inputs[..., 1], sdf_values, colors='k', levels=[0], linewidths=surface_width, **kwargs)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

def get_unique(df, key):
    vals = df[key].unique()
    if len(vals) != 1:
        raise ValueError(f"Expected one {key}, got {vals}")
    return vals[0]


def get_basis(mu0, mu1, mu2, preserve_angle=True, preserve_norm=True):
    u1 = mu1 - mu0
    u2 = mu2 - mu0
    if preserve_angle:
        u2 -= u1 * (u1 @ u2) / (u1 @ u1)

    if preserve_norm:
        u2 *= np.linalg.norm(u1) / np.linalg.norm(u2)

    x = np.linalg.lstsq(np.hstack([u1[:, None], u2[:, None]]), np.hstack([(mu1 - mu0)[:, None], (mu2 - mu0)[:, None]]))[0]
    x1, x2 = x[:, 0], x[:, 1]

    return u1, u2, x1, x2

def tabilize(results, precisions, rank_order, suffixes=None, hlines = []):

  def rankify(x, order):
    # Turn a vector of values into a list of ranks, while handling ties.
    assert len(x.shape) == 1
    if order == 0:
      return np.full_like(x, 1e5, dtype=np.int32)
    u = np.sort(np.unique(x))
    if order == 1:
      u = u[::-1]
    r = np.zeros_like(x, dtype=np.int32)
    for ui, uu in enumerate(u):
      mask = x == uu
      r[mask] = ui
    return np.int32(r)

  names = results.keys()
  data = np.array(list(results.values()))
  assert len(names) == len(data)
  data = np.array(data)

  tags = [' \cellcolor{tabfirst}',
          '\cellcolor{tabsecond}',
          ' \cellcolor{tabthird}',
          '                     ']

  max_len = max([len(v) for v in list(names)])
  names_padded = [v + ' '*(max_len-len(v)) for v in names]

  data_quant = np.round((data * 10.**(np.array(precisions)[None, :]))) / 10.**(np.array(precisions)[None, :])
  if suffixes is None:
    suffixes = [''] * len(precisions)

  tagranks = []
  for d in range(data_quant.shape[1]):
    tagranks.append(np.clip(rankify(data_quant[:,d], rank_order[d]), 0, len(tags)-1))
  tagranks = np.stack(tagranks, -1)

  for i_row in range(len(names)):
    line = '\t'
    if i_row in hlines:
      line += '\\hline\n'
    line += names_padded[i_row]
    for d in range(data_quant.shape[1]):
      line += ' & '
      if rank_order[d] != 0 and not np.isnan(data[i_row,d]):
        line += tags[tagranks[i_row, d]]
      if np.isnan(data[i_row,d]):
        line += ' - '
      else:
        assert precisions[d] >= 0
        line += ('{:' + f'0.{precisions[d]}f' + '}').format(data_quant[i_row,d]) + suffixes[d]
    line += ' \\\\'
    print(line)