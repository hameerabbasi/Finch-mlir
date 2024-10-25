// RUN: finch-opt %s | finch-opt | FileCheck %s
//./bin/finch-opt ../test13.mlir --loop-invariant-code-motion --inline --finch-simplifier  --finch-instantiate --finch-looplet-pass --finch-simplifier --finch-instantiate --finch-looplet-pass --finch-simplifier --sparsifier |  mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/Users/jaeyeonwon/llvm-project/build/lib/libmlir_runner_utils.dylib,/Users/jaeyeonwon/llvm-project/build/lib/libmlir_c_runner_utils.dylib
#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
module {

    func.func private @printMemrefF32(%ptr:memref<*xf32>) attributes {llvm.emit_c_interface}
    func.func private @printMemrefInd(%ptr:memref<*xindex>) attributes {llvm.emit_c_interface}

    func.func @buffers_from_sparsevector(%jump : index) -> (memref<?xindex>, memref<?xindex>, memref<?xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %f1 = arith.constant 1.0 : f32

      %6 = tensor.empty() : tensor<1024xf32, #SparseVector>
      %7 = scf.for %i = %c0 to %c8 step %c1 iter_args(%vin = %6) -> tensor<1024xf32, #SparseVector> {
        %ii = arith.muli %i, %jump : index
        %vout = tensor.insert %f1 into %vin[%ii] : tensor<1024xf32, #SparseVector>
        scf.yield %vout : tensor<1024xf32, #SparseVector>
      }
      %8 = sparse_tensor.load %7 hasInserts : tensor<1024xf32, #SparseVector>
      sparse_tensor.print %8 : tensor<1024xf32, #SparseVector>
      %9 = sparse_tensor.positions %8 {level = 0 :index} : tensor<1024xf32, #SparseVector> to memref<?xindex>
      %10 = sparse_tensor.coordinates %8 {level = 0 :index} : tensor<1024xf32, #SparseVector> to memref<?xindex>
      %11 = sparse_tensor.values %8 : tensor<1024xf32, #SparseVector> to memref<?xf32>
      return %9,%10,%11: memref<?xindex>, memref<?xindex>, memref<?xf32>
    }


    func.func private @dense_level(%pos: index, %shape: index) -> !finch.looplet {
      %l0 = finch.lookup
        body = {
          ^bb(%idx : index) :
            %0 = arith.muli %pos, %shape : index
            %1 = arith.addi %0, %idx : index
            %2 = finch.nextlevel %1 : (index) -> (!finch.looplet)
            finch.return %2 : !finch.looplet
        }
      return %l0 : !finch.looplet
    }


    func.func private @sparse_level(%pos: index, %ptr:memref<?xindex>, %crd:memref<?xindex>) -> !finch.looplet {
      %fp_0 = arith.constant 0.0 : f32
      %fp_1 = arith.constant 1.0 : f32
      %c1 = arith.constant 1 : index
     
      %l0 = finch.stepper  
          seek={
            ^bb0(%idx : index):
              %firstpos = func.call @binarysearch(%pos, %idx, %ptr, %crd) : (index, index, memref<?xindex>, memref<?xindex>) -> (index) 
              finch.return %firstpos : index
          }
          stop={
            ^bb(%pos1 : index):
              %currcrd = memref.load %crd[%pos1] : memref<?xindex>
              %stopub = arith.addi %currcrd, %c1 : index
              finch.return %stopub : index
          } 
          body={
            ^bb(%pos1 : index):
              %currcrd = memref.load %crd[%pos1] : memref<?xindex>

              %zero_run = finch.run %fp_0 : (f32) -> (!finch.looplet)
              %nonzero_run = finch.nextlevel %pos1 : (index) -> (!finch.looplet)
              %seq = finch.sequence %currcrd, %zero_run, %nonzero_run : (index, !finch.looplet, !finch.looplet) -> (!finch.looplet)
              finch.return %seq : !finch.looplet
          } 
          next={
            ^bb0(%pos1 : index):
              %nextpos = arith.addi %pos1, %c1 : index
              finch.return %nextpos : index 
          }

      // nextpos = pos+1
      // nextoffset = ptr[pos+1]
      // curroffset = ptr[pos]
      // empty = curroffset == nextoffset
      // if (empty) {
      //   return 0 
      // } else {
      //   lastoffset = nextoffset - 1
      //   last_nnz_crd = crd[lastoffset]
      //   last_nnz_ub = last_nnz_crd + 1
      //   return last_nnz_ub
      // }
      %nextpos = arith.addi %pos, %c1 : index
      %nextoffset = memref.load %ptr[%nextpos] : memref<?xindex>
      %curroffset = memref.load %ptr[%pos] : memref<?xindex>
      %empty = arith.cmpi eq, %curroffset, %nextoffset : index
      %zero_ub = scf.if %empty -> (index) {
        %c0 = arith.constant 0 : index
        scf.yield %c0 : index
      } else {
        %lastoffset = arith.subi %nextoffset, %c1 : index
        %last_nnz_crd = memref.load %crd[%lastoffset] : memref<?xindex>
        %last_nnz_ub = arith.addi %last_nnz_crd, %c1 : index
        scf.yield %last_nnz_ub : index
      }

      %zero_run = finch.run %fp_0 : (f32) -> (!finch.looplet)
      %l1 = finch.sequence %zero_ub, %l0, %zero_run : (index, !finch.looplet, !finch.looplet) -> (!finch.looplet)

      return %l1 : !finch.looplet
    }


    func.func private @element_level(%pos: index, %val : memref<?xf32>) -> !finch.looplet {
      %currval = memref.load %val[%pos] : memref<?xf32>
      %run = finch.run %currval : (f32) -> (!finch.looplet)
      return %run : !finch.looplet
    }

    func.func private @binarysearch(%pos: index, %idx : index, %ptr : memref<?xindex>, %crd : memref<?xindex>) -> index {
      // i = ptr[pos];
      // while(i<ptr[pos+1] && crd[i] < idx) {
      //   i += 1;
      // }

      %c1 = arith.constant 1 : index
      %offset = memref.load %ptr[%pos] : memref<?xindex>

      %nextpos = arith.addi %pos, %c1 : index
      %nextoffset = memref.load %ptr[%nextpos] : memref<?xindex>
      
      %search = scf.while (%i = %offset) : (index) -> (index) {
        %cmp1 = arith.cmpi ult, %i, %nextoffset : index 
        %cmp2 = scf.if %cmp1 -> (i1) {
          %currcrd = memref.load %crd[%i] : memref<?xindex>
          %cmp = arith.cmpi ult, %currcrd, %idx : index 
          scf.yield %cmp : i1
        } else {
          %false = arith.constant 0 : i1
          scf.yield %false : i1
        }
        scf.condition(%cmp2) %i : index
      } do {
        ^bb(%i:index) :
          %next = arith.addi %i, %c1 : index
          scf.yield %next : index
      }
      return %search : index 
    }

    func.func @main()  {
      // sparse_tensor -> extract memref (pos,crd,val) from sparse_Tensor -> build looplet representation using those memrefs
      // -> perform computation with finch dialect -> lower finch loop with finch passes to llvm -> run llvm

      /////////////////////////////////
      // Defining 2D Tensor with Looplet      
      /////////////////////////////////

      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %buff:3 = call @buffers_from_sparsevector(%c2) : (index) -> (memref<?xindex>, memref<?xindex>, memref<?xf32>)
      %buff2:3 = call @buffers_from_sparsevector(%c3) : (index) -> (memref<?xindex>, memref<?xindex>, memref<?xf32>)

      %ptr = memref.cast %buff#0 :  memref<?xindex> to memref<?xindex>
      %crd = memref.cast %buff#1 :  memref<?xindex> to memref<?xindex>
      %val = memref.cast %buff#2 :  memref<?xf32> to memref<?xf32>
      %ptr2= memref.cast %buff2#0 :  memref<?xindex> to memref<?xindex>
      %crd2= memref.cast %buff2#1 :  memref<?xindex> to memref<?xindex>
      %val2= memref.cast %buff2#2 :  memref<?xf32> to memref<?xf32>
      

      /////////////////////////////////
      // Wrap memrefs to Looplets
      /////////////////////////////////

      %ALvl0 = finch.definelevel {
        ^bb0(%pos : index) :
          %l = func.call @sparse_level(%pos, %ptr, %crd): (index, memref<?xindex>, memref<?xindex>) -> !finch.looplet
          finch.return %l : !finch.looplet
      }

      %ALvl1 = finch.definelevel {
        ^bb(%pos:index):
          %l = func.call @element_level(%pos, %val): (index, memref<?xf32>) -> !finch.looplet
          finch.return %l : !finch.looplet
      }

      %BLvl0 = finch.definelevel {
        ^bb0(%pos : index) :
          %l = func.call @sparse_level(%pos, %ptr2, %crd2): (index, memref<?xindex>, memref<?xindex>) -> !finch.looplet
          finch.return %l : !finch.looplet
      }

      %BLvl1 = finch.definelevel {
        ^bb(%pos:index):
          %l = func.call @element_level(%pos, %val2): (index, memref<?xf32>) -> !finch.looplet
          finch.return %l : !finch.looplet
      }


      %b0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %shape = arith.constant 20 : index
      %val3 = memref.alloc(%shape) : memref<?xf32>                                           
      %fp_m1 = arith.constant -1.0 : f32
      
      scf.for %j = %b0 to %shape step %c1 {                                           
        memref.store %fp_m1, %val3[%j] : memref<?xf32>                                      
      }
      
      %CLvl0 = finch.definelevel {
        ^bb0(%pos : index) :
          %l = func.call @dense_level(%pos, %shape): (index, index) -> !finch.looplet
          finch.return %l : !finch.looplet
      }

      %CLvl1 = finch.definelevel {
        ^bb(%pos:index):
          %l = func.call @element_level(%pos, %val3): (index, memref<?xf32>) -> !finch.looplet
          finch.return %l : !finch.looplet
      }

      ///////////////////////////////////
      //// Main Code
      ///////////////////////////////////
       

      %b1 = arith.constant 10 : index

      %l0a = finch.getlevel %ALvl0, %c0 : (!finch.looplet, index) -> (!finch.looplet)
      %l0b = finch.getlevel %BLvl0, %c0 : (!finch.looplet, index) -> (!finch.looplet)
      %l0c = finch.getlevel %CLvl0, %c0 : (!finch.looplet, index) -> (!finch.looplet)
      
      scf.for %j = %b0 to %b1 step %c1 {                                           
        %p1a = finch.access %l0a, %j : index                              
        %p1b = finch.access %l0b, %j : index                              
        %p1c = finch.access %l0c, %j : index                              

        %l1a = finch.getlevel %ALvl1, %p1a : (!finch.looplet, index) -> (!finch.looplet)
        %l1b = finch.getlevel %BLvl1, %p1b : (!finch.looplet, index) -> (!finch.looplet)
        %l1c = finch.getlevel %CLvl1, %p1c : (!finch.looplet, index) -> (!finch.looplet)


        %va = finch.access %l1a, %j : f32                                           
        %vb = finch.access %l1b, %j : f32                                           
        %vc = finch.access %l1c, %j : f32                                           


        %vab = arith.mulf %va, %vb : f32   
        finch.assign %vc = %vab : f32
        //finch.assign %vc = %va : f32
      }
   
      // Print %sum
      %z = memref.cast %val3 :  memref<?xf32> to memref<*xf32>
      call @printMemrefF32(%z): (memref<*xf32>) -> ()
      
      return 
    }
}
