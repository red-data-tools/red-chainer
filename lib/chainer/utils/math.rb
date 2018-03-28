module Chainer
  module Utils
    module Math
      def self.tensordot(a, b, axes)
        if axes.is_a?(Integer)
          axes_a = (-axes...0).to_a
          axes_b = (0...axes).to_a
        else axes.is_a?(Array)
          axes_a, axes_b = axes
        end

        axes_a = Array(axes_a)
        axes_b = Array(axes_b)
        na = axes_a.size
        nb = axes_b.size

        as = a.shape
        nda = a.ndim
        bs = b.shape
        ndb = b.ndim
        equal = true
        if na != nb
            equal = false
        else
          na.times do |k|
            if as[axes_a[k]] != bs[axes_b[k]]
              equal = false
              break
            end

            if axes_a[k] < 0
              axes_a[k] += nda
            end

            if axes_b[k] < 0
              axes_b[k] += ndb
            end
          end
        end

        raise "shape-mismatch for sum" unless equal

        notin = (0...nda).reject { |i| axes_a.include?(i) } 
        newaxes_a = notin + axes_a
        n2 = 1
        axes_a.each do |axis|
          n2 *= as[axis]
        end
        tmp = a.shape.reduce(:*) / n2
        newshape_a = [tmp, n2]
        olda = notin.map { |axis| as[axis] }

        notin = (0...ndb).reject { |i| axes_b.include?(i) }
        newaxes_b = axes_b + notin
        n2 = 1
        axes_b.each do |axis|
          n2 *= bs[axis]
        end
        tmp = b.shape.reduce(:*) / n2
        # newshape_b = [n2, -1]
        newshape_b = [n2, tmp]
        oldb = notin.map { |axis| bs[axis] }

        at = a.transpose(*newaxes_a).reshape(*newshape_a)
        bt = b.transpose(*newaxes_b).reshape(*newshape_b)
        res = at.dot(bt)
       

        return res.reshape(*(olda + oldb))
      end
    end
  end
end

