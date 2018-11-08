module Chainer
  module Utils
    module Conv
      def self.get_conv_outsize(size, k, s, p, cover_all: false, d: 1)
        dk = k + (k - 1) * (d - 1)
        if cover_all
          (size + p * 2 - dk + s - 1).div(s) + 1
        else
          (size + p * 2 - dk).div(s) + 1
        end
      end

      def self.im2col(img, kh, kw, sy, sx, ph, pw, pval: 0, cover_all: false, dy: 1, dx: 1)
        n, c, h, w = img.shape

        out_h = self.get_conv_outsize(h, kh, sy, ph, cover_all: cover_all, d: dy)
        raise 'Height in the output should be positive.' if out_h <= 0
        out_w = self.get_conv_outsize(w, kw, sx, pw, cover_all: cover_all, d: dx)
        raise 'Width in the output should be positive.' if out_w <= 0

        # padding
        # TODO: ref: numpy.pad
        pad_bottom = ph + sy - 1
        pad_right = pw + sx - 1
        pad_img = img.class.new(n, c, (h + ph + pad_bottom), (w + pw + pad_right)).fill(pval)
        pad_img[nil, nil, ph...(ph+h), pw...(pw+w)] = img

        col = pad_img.class.new(n, c, kh, kw, out_h, out_w).rand(1)

        kh.times do |j|
          jdy = j * dy
          j_lim = [jdy + sy * out_h, pad_img.shape[2]].min
          kw.times do |i|
            idx = i * dx
            i_lim = [idx + sx * out_w, pad_img.shape[3]].min
            col[nil, nil, j, i, nil, nil] = pad_img[nil, nil, (jdy...j_lim).step(sy), (idx...i_lim).step(sx)]
          end
        end

        col
      end

      def self.col2im(col, sy, sx, ph, pw, h, w, dy: 1, dx: 1)
        n, c, kh, kw, out_h, out_w = col.shape
        img = col.class.zeros(n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1)
        kh.times do |j|
          jdy = j * dy
          j_lim = [jdy + sy * out_h, img.shape[2]].min
          kw.times do |i|
            idx = i * dx
            i_lim = [idx + sx * out_w, img.shape[3]].min
            img[nil, nil, (jdy...j_lim).step(sy), (idx...i_lim).step(sx)] += col[nil, nil, j, i, nil, nil]
          end
        end
        return img[nil, nil, (ph...(h + ph)), (pw...(w + pw))]
      end
    end
  end
end
