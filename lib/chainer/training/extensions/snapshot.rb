module Chainer
  module Training
    module Extensions
      class Snapshot < Extension
        attr_accessor :save_class, :filename_proc, :target

        def self.snapshot_object(target:, save_class:, &block)
          self.new(save_class: save_class, filename_proc: block, target: target)
        end

        def self.snapshot(save_class: nil, &block)
          self.new(save_class: save_class, filename_proc: block)
        end

        def initialize(save_class: nil, filename_proc: nil, target: nil)
          @save_class = save_class || Chainer::Serializers::MarshalSerializer
          @filename_proc = filename_proc || Proc.new { |trainer| "snapshot_iter_#{trainer.updater.iteration}" }
          @target = target
        end

        def call(trainer)
          target = @target || trainer
          filename = filename_proc.call(trainer)
          prefix = "tmp#{filename}"
          temp_file = Tempfile.create(basename: prefix, tmpdir: trainer.out)
          save_class.save_file(temp_file, trainer)
          FileUtils.move(temp_file.path, File.join(trainer.out, filename))
        end
      end
    end
  end
end

