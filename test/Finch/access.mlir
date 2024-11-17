// RUN: finch-opt %s | finch-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        %1 = index.castu %0 : i32 to index
       
        %val = arith.constant 3.0 : f32
        // CHECK: %{{.*}} = finch.run %{{.*}} : (f32) -> (!finch.looplet)
        %3 = finch.run %val : (f32) -> (!finch.looplet)

        // CHECK: %{{.*}} = finch.access %{{.*}}, %{{.*}} : f32
        %2 = finch.access %3, %1 : f32 
 
        return
    }
}
