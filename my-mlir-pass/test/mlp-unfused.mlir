#matrix = affine_map<(d0, d1) -> (d0, d1)>
#bias = affine_map<(d0, d1) -> (d1)>

module 
{
  func.func @mlp(%X:  memref<64x64xf32>, %W1: memref<64x64xf32>, %B1: memref<64xf32>, %W2: memref<64x64xf32>, %B2: memref<64xf32>, %Y:  memref<64x64xf32>) attributes {llvm.emit_c_interface} 
{

        %zero = arith.constant 0.0 : f32

        // Intermediate matrices.
        %T1 = memref.alloc() : memref<64x64xf32>
        %T2 = memref.alloc() : memref<64x64xf32>
        %H  = memref.alloc() : memref<64x64xf32>
        %T3 = memref.alloc() : memref<64x64xf32>

        // 1. T1 = X × W1
        linalg.fill ins(%zero : f32) outs(%T1 : memref<64x64xf32>)
        linalg.matmul ins(%X, %W1 : memref<64x64xf32>, memref<64x64xf32>) outs(%T1 : memref<64x64xf32>)

        // 2. T2[i,j] = T1[i,j] + B1[j]
        linalg.generic {indexing_maps = [#matrix, #bias, #matrix], iterator_types = ["parallel", "parallel"]} 
        ins(%T1, %B1 : memref<64x64xf32>, memref<64xf32>)
        outs(%T2 : memref<64x64xf32>) 
        {
            ^bb0(%value: f32, %biasValue: f32, %unused: f32):
            %sum = arith.addf %value, %biasValue : f32
            linalg.yield %sum : f32
        }

        // 3. H[i,j] = ReLU(T2[i,j])
        linalg.generic { indexing_maps = [#matrix, #matrix], iterator_types = ["parallel", "parallel"] } 
        ins(%T2 : memref<64x64xf32>)
        outs(%H : memref<64x64xf32>) 
        {
            ^bb0(%value: f32, %unused: f32):
                %result = arith.maximumf %value, %zero : f32
                linalg.yield %result : f32
        }

        // 4. T3 = H × W2
        linalg.fill ins(%zero : f32) outs(%T3 : memref<64x64xf32>)
        linalg.matmul ins(%H, %W2 : memref<64x64xf32>, memref<64x64xf32>) outs(%T3 : memref<64x64xf32>)

        // 5. Y[i,j] = T3[i,j] + B2[j]
        linalg.generic { indexing_maps = [#matrix, #bias, #matrix], iterator_types = ["parallel", "parallel"]} 
        ins(%T3, %B2 : memref<64x64xf32>, memref<64xf32>)
        outs(%Y : memref<64x64xf32>) 
        {
            ^bb0(%value: f32, %biasValue: f32, %unused: f32):
                %sum = arith.addf %value, %biasValue : f32
                linalg.yield %sum : f32
        }

        memref.dealloc %T1 : memref<64x64xf32>
        memref.dealloc %T2 : memref<64x64xf32>
        memref.dealloc %H  : memref<64x64xf32>
        memref.dealloc %T3 : memref<64x64xf32>

        //return
        return

  }

}