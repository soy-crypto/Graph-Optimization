#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <string>
#include <cstdint>
#include <sstream>
#include <utility>

/*custom head file */
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

//name space 
namespace
{
    //pass - Print
    struct PrintOpsPass : PassWrapper<PrintOpsPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpsPass);

        StringRef getArgument() const override
        {
            return "print-ops";
        }

        void runOnOperation() override
        {
            getOperation()->walk([](arith::AddIOp op) 
            { 
                llvm::outs() << "ADD: " << op << "\n"; 
            });


            getOperation()->walk([](arith::MulIOp op) 
            {
                llvm::outs() << "MUL: " << op << "\n"; 
            });
              
        }

    };

    //pass - Function
    struct FunctionStatsPass : PassWrapper<FunctionStatsPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FunctionStatsPass);

        StringRef getArgument() const override
        {
            return "func-stats";
        }

        StringRef getDescription() const override
        {
            return "Print function statistics";
        }

        void runOnOperation() override
        {
            getOperation()->walk([](func::FuncOp func)
            {
                //show func
                llvm::outs() << "-----" << func.getName() << "-----\n"; 
                //Compute
                llvm::StringMap<int> freqMap;
                func.walk([&](Operation *op)
                {
                    if(!isa<func::FuncOp>(op))
                    {
                        freqMap[op->getName().getStringRef()]++;
                    }

                });

                //Output
                llvm::outs() << "Function: " << func.getName() << "\n";
                llvm::outs() << "Content: \n" << func << "\n";
                llvm::outs() << "Arguments: "<< func.getNumArguments() << "\n";

                for(auto arg : func.getArguments())
                {
                    llvm::outs() << "Arg Type: " << arg.getType() << "\n";
                }

                llvm::outs() << "Operations: \n";
                for (auto &it : freqMap)
                {
                    llvm::outs() << it.getKey() << " : " << it.getValue() << "\n";
                }

                for (Type t : func.getResultTypes())
                {
                    llvm::outs() << "Return Type: " << t << "\n";
                }
                
                //LOOP 
                int loopCount = 0;
                func.walk([&](scf::ForOp loop)
                {
                    //count number of loops
                    loopCount++;

                    //output parameters
                    llvm::outs() << "Found loop \n";
                    llvm::outs() << "Lower Bound: " << loop.getLowerBound() << "\n";
                    llvm::outs() << "Upper Bound: " << loop.getUpperBound() << "\n";
                    llvm::outs() << "Step: " << loop.getStep() << "\n";
                    llvm::outs() << loop << "\n";
                    
                    //inside loop body
                    llvm::StringMap<int> counts;
                    loop.getBody()->walk([&](Operation *op)
                    {
                        if(!isa<scf::YieldOp>(op))
                        {
                            counts[op->getName().getStringRef()]++;
                        }

                    });

                    for(auto &item : counts)
                    {
                        llvm::outs() << item.getKey() << " : " << item.getValue() << "\n";
                    }

                });

                bool optimized = false;
                func.walk([&](scf::ForOp loop)
                {
                    loop.getBody()->walk([&](arith::AddIOp add)
                    {
                        bool invariant = true;
                        for(Value operand : add.getOperands())
                        {
                            auto *defOp = operand.getDefiningOp();
                            if(defOp && loop->isAncestor(defOp))
                            {
                                invariant = false;
                                break;
                            }

                        }//

                        if(invariant == true)
                        {
                            llvm::outs() << "Found invariant: " << add << "\n";
                            add->moveBefore(loop);
                            optimized = true;
                        }

                    });

                });
                
                llvm::outs() << "Loop count " << loopCount << "\n";

                //Optimized function
                if(optimized == true)
                {
                    llvm::outs() << "Optimized Func :" << func << "\n";
                }
                
                llvm::outs() << "\n\n";

            });

        }

    };

    //pass - LICM
    struct LICMPass : PassWrapper<LICMPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LICMPass);

        StringRef getArgument() const override
        {
            return "licm";
        }

        StringRef getDescription() const override
        {
            return "licm code elimination!";
        }

        void runOnOperation() override
        {
            //Disply start
            llvm::outs() << "-----LICM start -----" << "\n";

            //travers all functions
            getOperation()->walk([&](func::FuncOp func)
            {
                //Traverse all loops
                bool funcChanged = false;
                func.walk([&](scf::ForOp loop)
                {
                    //current loop
                    bool moved = true;
                    while(true)
                    {
                        //TC
                        if(moved == false)
                        {
                            break;
                        }

                        //Body
                        SmallVector<Operation*> movedOps;
                        loop.getBody()->walk([&](Operation *op)
                        {
                            //Check current op
                            bool found = false;
                            if(isa<arith::ConstantOp>(op))
                            {
                                found = true;
                            }
                            else if(!isa<scf::YieldOp>(op) && !isa<func::ReturnOp>(op))
                            {
                                bool flag = true;
                                for(Value operand : op->getOperands())
                                {
                                    if(operand != loop.getInductionVar())
                                    {
                                        auto *defOp = operand.getDefiningOp();
                                        flag &= defOp && loop->isAncestor(defOp) ? false : true; 
                                    }
                                    else
                                    {
                                        flag = false;
                                        break;
                                    }
                                    
                                }//for

                                //Update found
                                found = flag;

                            }//else

                            //Update movedOps
                            if(found == true)
                            {
                                movedOps.push_back(op);
                            }

                        });
                        
                        //Move operations
                        if(!movedOps.empty())
                        {
                            for(Operation *op : movedOps)
                            {
                                op->moveBefore(loop);
                            }//

                            moved = true;
                            funcChanged = true;

                        }//if
                        else
                        {
                            moved = false;
                        }
                        
                    }//while
                    
                });

                //Display optimized func
                if(funcChanged == true)
                {
                    llvm::outs() << "Optimized! : " << func << "\n";
                }

            });
            
            //Display end
            llvm::outs() << "-----LICM end -----\n" << "\n";

            //return
            return;

        }//void

    }; //LICM Pass

    //pass - DCE
    struct DCEPass : PassWrapper<DCEPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DCEPass);
        
        StringRef getArgument() const override
        {
            return "dce";
        }

        StringRef getDescription() const override
        {
            return "dce elimination!";
        }
        
        void runOnOperation() override
        {
            //Display start
            llvm::outs() << "-----DCE Start-----" << "\n";

            //Travers all ops
            getOperation()->walk([](func::FuncOp func)
            {
                //Dec found status
                bool found = false;

                //Dec checking
                bool moved = true;
                SmallVector<Operation*> movedOps;
                while(true)
                {
                    //tc
                    if(moved == false)
                    {
                        break;
                    }

                    //body
                    func.walk([&](Operation* op)
                    {
                        if(op->use_empty() && op->getNumResults() >= 1)
                        {
                            movedOps.push_back(op);
                            found = true;
                        }

                    });

                    //moved
                    if(!movedOps.empty())
                    {
                        for(Operation* op : movedOps)
                        {
                            op->erase();
                        }
                        movedOps.clear();
                        
                        moved = true;
                    }
                    else
                    {
                        moved = false;
                    }
                    
                }//while

                //show dced func
                if(found == true)
                {
                    llvm::outs() << "DECed Func : " << func << "\n";
                }
                
            });

            //Display end
            llvm::outs() << "-----DCE End-----" << "\n";
            
            //Return
            return;
        }

    }; //DCE Pass

    //pass - CF
    struct CFPass: PassWrapper<CFPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CFPass);
        
        StringRef getArgument() const override
        {
            return "cf";
        }

        StringRef getDescription() const override
        {
            return "cf elimination";
        }

        //get all dead ops of current func
        llvm::DenseMap<Operation*, int64_t> getFoldedOps(func::FuncOp func)
        {
            //Check validity
            if(func == nullptr)
            {
                return llvm::DenseMap<Operation*, int64_t>();
            }
            
            //Get all deadOps of current funct
            llvm::DenseMap<Operation*, int64_t> foldedOpsMap;
            func.walk([&foldedOpsMap, this](Operation* op)
            {
                //Init
                if(op->getNumResults() != 1 || op->getOperands().size() != 2)
                {
                    return; 
                }
                
                //Check current op is dead op
                auto operands = op->getOperands();
                auto lhsDef = operands[0].getDefiningOp<arith::ConstantOp>();
                auto rhsDef = operands[1].getDefiningOp<arith::ConstantOp>();
                if(lhsDef != nullptr && rhsDef != nullptr)
                {
                    //get left attrs and right attrs
                    auto lhsAttr = dyn_cast<IntegerAttr>(lhsDef.getValue());
                    auto rhsAttr = dyn_cast<IntegerAttr>(rhsDef.getValue());
                    if(lhsAttr == nullptr || rhsAttr == nullptr)
                    {
                        return;
                    }

                    //compute new constant
                    int64_t lV = lhsAttr.getInt(), rV = rhsAttr.getInt();
                    if(isa<arith::AddIOp>(op) || isa<arith::MulIOp>(op) || isa<arith::SubIOp>(op) || (isa<arith::RemSIOp>(op) && rV != 0) || (isa<arith::DivSIOp>(op) && rV != 0))
                    {
                        foldedOpsMap[op] = this->getConstantResult(op, lV, rV);
                    }

                }//if

                //Return
                return;
            });

            //Return
            return foldedOpsMap;
        
        }//getDeadOps()

        int64_t getConstantResult(Operation* op, int64_t lV, int64_t rV)
        {
            //Check validity
            if(op == nullptr)
            {
                return 0;
            }

            //Compoute
            int64_t result = 0;
            if(isa<arith::AddIOp>(op))
            {
                result = lV + rV;
            }
            else if(isa<arith::MulIOp>(op))
            {
                result = lV * rV;
            }
            else if(isa<arith::SubIOp>(op))
            {
                result = lV - rV;
            }
            else if(isa<arith::DivSIOp>(op))
            {
                result = lV / rV;
            }
            else // % computatiob
            {
                result = lV % rV;
            }

            //Return
            return result;
        }//

        //operations
        void runOnOperation() override
        {   
            //traverse all funcs
            getOperation()->walk([this](func::FuncOp func)
            {
                //Check validity
                if(func == nullptr)
                {
                    return;
                }

                //CF perform
                bool deleted = true;
                llvm::DenseMap<Operation* , int64_t> foldedOpsMap;
                while(true)
                {
                    if(deleted == false)
                    {
                        break;
                    }

                    //get dead ops of current func
                    foldedOpsMap = this->getFoldedOps(func);
                    
                    //batch erase dead ops
                    if(!foldedOpsMap.empty())
                    {
                        for(auto &[op, result] : foldedOpsMap)
                        {
                            //replace CF ops
                            OpBuilder builder(op);
                            auto newConst = builder.create<arith::ConstantIntOp>(op->getLoc(), op->getResult(0).getType(), result);
                            op->getResult(0).replaceAllUsesWith(newConst.getResult());
                            
                            //erase op
                            op->erase();
                        }//

                        //update flag delete
                        deleted = true;
                    }
                    else
                    {
                        deleted = false;
                    }

                }//while

                //Return
                return;

            }); //getOperation()

            //Return
            return;
        }//void

    }; // CF pass

    //pass - CSE
    struct CSEPass : PassWrapper<CSEPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSEPass);
        
        StringRef getArgument() const override
        {
            return "toy-cse";
        }

        StringRef getDescription() const override
        {
            return "csed elimination!";
        }

        //operation
        void runOnOperation() override
        {
            getOperation()->walk([this](func::FuncOp func)
            {
                //checking validity
                if(func == nullptr)
                {
                    return;
                }

                //cse deletion
                llvm::DenseMap<Operation*, Operation*> cseOps;
                bool deleted = true;
                while(true)
                {
                    //tc
                    if(deleted == false)
                    {
                        break;
                    }

                    //get cse ops
                    cseOps = this->getCseOps(func);

                    //delete cse code
                    if(!cseOps.empty())
                    {
                        for(auto &[op1, op2] : cseOps)
                        {
                            op1->getResult(0).replaceAllUsesWith(op2->getResult(0));
                            op1->erase();
                        }

                        deleted = true;

                    }//
                    else
                    {
                        deleted = false;
                    }

                }//while
                

            });//

            //Return
            return;
        }//void

        //get cse ops
        llvm::DenseMap<Operation*, Operation*> getCseOps(func::FuncOp func)
        {
            //check validity
            if(func == nullptr)
            {
                return llvm::DenseMap<Operation*, Operation*>();
            }

            //get cse ops
            llvm::DenseMap<Operation*, Operation*> cseOps;
            llvm::StringMap<Operation*> firstOps;
            func->walk([&firstOps, &cseOps](Operation* op)
            {
                //checking validity
                if(op == nullptr || op->getNumOperands() != 2)
                {
                    return;
                }

                //get cse ops
                std::string opName = op->getName().getStringRef().str();

                auto operands = op->getOperands();
                std::ostringstream s1, s2;
                s1 << opName << "-" << operands[0].getAsOpaquePointer() << "-" << operands[1].getAsOpaquePointer();
                s2 << opName << "-" << operands[1].getAsOpaquePointer() << "-" << operands[0].getAsOpaquePointer();
                std::string key1 = opName + "-" + s1.str();
                std::string key2 = opName + "-" + s2.str();

                if(isa<arith::AddIOp>(op) || isa<arith::MulIOp>(op))
                {
                    if(firstOps.contains(key1))
                    {
                        cseOps[op] = firstOps[key1];
                    }
                    else if(firstOps.contains(key2))
                    {
                        cseOps[op] = firstOps[key2];
                    }
                    else
                    {
                        firstOps[key1] = op;
                        firstOps[key2] = op;
                    }

                }
                else if(isa<arith::SubIOp>(op) || isa<arith::DivSIOp>(op) || isa<arith::RemSIOp>(op))
                {
                    if(firstOps.contains(key1))
                    {
                        cseOps[op] = firstOps[key1];
                    }
                    else
                    {
                        firstOps[key1] = op;
                    }

                }

                //Return
                return;
            });

            //Return
            return cseOps;

        }//get cse ops
 
    };// Pass

    //pass - IC(Instant Combine)
    struct ICPass : PassWrapper<ICPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ICPass);

        StringRef getArgument() const override
        {
            return "ic";
        }

        StringRef getDescription() const override
        {
            return "ic deletion";
        }

        void runOnOperation() override
        {
            getOperation()->walk([this](func::FuncOp func)
            {
                //check validity
                if(func == nullptr)
                {
                    return;
                }

                //delete zero IC ops
                bool deleted = true;
                while(true)
                {
                    //tc
                    if(deleted == false)
                    {
                        break;
                    }

                    //delete icOps
                    deleted = this->deleteICOps(this->getZeroICOps(func));

                }//

                //delete one IC ops
                deleted = true;
                while(true)
                {
                    //tc
                    if(deleted == false)
                    {
                        break;
                    }

                    //delete icOps
                    deleted = this->deleteICOps(this->getOneICOps(func));

                }//

                //Return
                return;

            });

        }

        llvm::DenseMap<Operation*, Value> getZeroICOps(func::FuncOp func)
        {
            //check validity
            if(func == nullptr)
            {
                return llvm::DenseMap<Operation*, Value>();
            }

            //get IC Ops
            llvm::DenseMap<Operation*, Value> icOps;
            func.walk([this, &icOps](Operation* op)
            {
                //check validity
                if(op == nullptr || op->getNumOperands() != 2)
                {
                    return;
                }

                //get IC ops
                auto operands = op->getOperands();
                bool flag1 = isZero(operands[0]), flag2 = isZero(operands[1]);
                Value newValue;
                if(flag1 && !flag2)
                {
                    newValue = this->leftZeroCompute(op);
                }
                else if(!flag1 && flag2)
                {
                    newValue = this->rightZeroCompute(op);
                }
                else if(flag1 && flag2)
                {
                    newValue = operands[0];
                }
                
                if(newValue != nullptr)
                {
                    icOps[op] = newValue;
                }

                //return
                return;
            });

            //return
            return icOps;

        }//getZeroICOps

        llvm::DenseMap<Operation*, Value> getOneICOps(func::FuncOp func)
        {
            //check validity
            if(func == nullptr)
            {
                return llvm::DenseMap<Operation*, Value>();
            }

            //get One ICOps
            llvm::DenseMap<Operation*, Value> oneICOps;
            func.walk([this, &oneICOps](Operation* op)
            {
                //check validity
                if(op == nullptr || op->getNumOperands() != 2)
                {
                    return;
                }

                //check operands are one or not
                auto operands = op->getOperands();
                Value newValue;
                if(isOne(operands[0]) && isa<arith::MulIOp>(op))
                {
                    newValue = operands[1];
                }
                else if(isOne(operands[1]) && (isa<arith::MulIOp>(op) || isa<arith::DivSIOp>(op)))
                {
                    newValue = operands[0];
                }
                
                //update oneICOps
                if(newValue != nullptr)
                {
                    oneICOps[op] = newValue;
                }

                //Return
                return;
            });

            //return
            return oneICOps;

        }//getOneICOps

        bool deleteICOps(llvm::DenseMap<Operation*, Value> icOps)
        {
            //Check validity
            if(icOps.empty())
            {
                return false;
            }

            //update icOps first
            for(auto &[op1, v1] : icOps)
            {
                for(auto &[op2, v2] : icOps)
                {
                    if(v2 == op1->getResult(0))
                    {
                        v2 = v1;
                    }

                }//for

            }//for
            
            //delete ic ops
            for(auto &[op, variable] : icOps)
            {
                op->getResult(0).replaceAllUsesWith(variable);
                op->erase();
            }

            icOps.clear();

            //return
            return true;
        }

        bool isZero(Value operand)
        {
            //Check validity
            if(operand == nullptr || !operand.getDefiningOp<arith::ConstantOp>())
            {
                return false;
            }

            //Check
            bool flag = false;
            auto defOp = operand.getDefiningOp<arith::ConstantOp>();
            auto attr = dyn_cast<IntegerAttr>(defOp.getValue());
            if(attr && attr.getValue() == 0)
            {
                flag = true;
            }

            //Return
            return flag;
        }

        bool isOne(Value operand)
        {
            //check validity
            if(operand == nullptr || !operand.getDefiningOp<arith::ConstantOp>())
            {
                return false;
            }

            //check
            bool flag = false;
            auto defOp = operand.getDefiningOp<arith::ConstantOp>();
            auto attr = dyn_cast<IntegerAttr>(defOp.getValue());
            if(attr && attr.getValue() == 1)
            {
                flag = true;
            }

            //Return
            return flag;
        }

        Value leftZeroCompute(Operation* op)
        {
            //check validity
            if(op == nullptr)
            {
                return nullptr;
            }

            //compute
            Value result;
            auto operands = op->getOperands();
            if(isa<arith::AddIOp>(op))
            {
                result = operands[1];
            }
            else if(isa<arith::MulIOp>(op))
            {
                result = operands[0];
            }
            else if(isa<arith::DivSIOp>(op))
            {
                result = operands[0];
            }

            //Return
            return result;

        }//Value

        Value rightZeroCompute(Operation* op)
        {
            //check validity
            if(op == nullptr)
            {
                return nullptr;
            } 

            //compute
            auto operands = op->getOperands();
            Value result;
            if(isa<arith::AddIOp>(op))
            {
                result = operands[0];
            }
            else if(isa<arith::MulIOp>(op))
            {
                result = operands[1];
            }
            else if(isa<arith::SubIOp>(op))
            {
                result = operands[0];
            }

            //Return
            return result;
        }//

    }; // ICPass

    //pass - IC with pattern
    // x + 0 -> x
    struct AddZeroPattern : OpRewritePattern<arith::AddIOp>
    {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::AddIOp op, PatternRewriter &rewriter) const override
        {
            //get left and right defingin OPs and their attrs
            auto ldefOp = op.getLhs().getDefiningOp<arith::ConstantIntOp>();
            auto rdefOp = op.getRhs().getDefiningOp<arith::ConstantIntOp>();
            
            //check which zero patterns
            if(ldefOp && ldefOp.value() == 0)
            {
                rewriter.replaceOp(op, op.getRhs());
                return success();
            }
            else if(rdefOp && rdefOp.value() == 0)
            {
                rewriter.replaceOp(op, op.getLhs());
                return success();
            }

            //return
            return failure();

        }//

    };

    // x * 1 -> x
    struct MulOnePattern : OpRewritePattern<arith::MulIOp>
    {
        using OpRewritePattern:: OpRewritePattern;

        LogicalResult matchAndRewrite(arith::MulIOp op, PatternRewriter &rewriter) const override
        {
            auto lhs = op.getLhs(), rhs = op.getRhs();
            auto lcst = lhs.getDefiningOp<arith::ConstantIntOp>();
            auto rcst = rhs.getDefiningOp<arith::ConstantIntOp>();

            if(lcst && lcst.value() == 1)
            {
                rewriter.replaceOp(op, rhs);
                return success();
            }
            else if(rcst && rcst.value() == 1)
            {
                rewriter.replaceOp(op, lhs);
                return success();
            }

            //return
            return failure();

        }//logcial

    };

    // constant * constant -> constant
    struct ConstantFoldPattern : OpRewritePattern<arith::AddIOp>
    {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::AddIOp op, PatternRewriter &rewriter) const override
        {
            auto lhs = op.getLhs(), rhs = op.getRhs();
            auto lcst = lhs.getDefiningOp<arith::ConstantIntOp>();
            auto rcst = rhs.getDefiningOp<arith::ConstantIntOp>();

            if(lcst != nullptr && rcst != nullptr)
            {
                int64_t result = lcst.value() + rcst.value();
                auto newConst = rewriter.create<arith::ConstantIntOp>(op.getLoc(), result, 32);
                rewriter.replaceOp(op, newConst);
                return success();
            }
            else
            {
                return failure();
            }

        }

    };//

    // x * 0 -> 0
    struct MulZeroPattern : OpRewritePattern<arith::MulIOp>
    {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::MulIOp op, PatternRewriter &rewriter) const override
        {
            //get left oprand and right oprand
            auto lhs = op.getLhs(), rhs = op.getRhs();
            auto lcst = lhs.getDefiningOp<arith::ConstantIntOp>();
            auto rcst = rhs.getDefiningOp<arith::ConstantIntOp>();
            
            //check
            if(lcst != nullptr && lcst.value() == 0)
            {
                rewriter.replaceOp(op, lhs);
                return success();
            }
            else if(rcst != nullptr && rcst.value() == 0)
            {
                rewriter.replaceOp(op, rhs);
                return success();
            }
            else
            {
                return failure();
            }
            
        }

    }; 

    // *0 pattern
    struct SubSelfPattern : OpRewritePattern<arith::SubIOp>
    {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::SubIOp op, PatternRewriter &rewriter) const override
        {
            //get left and right operands
            auto lhs = op.getLhs(), rhs = op.getRhs();
            auto lcst = lhs.getDefiningOp<arith::ConstantIntOp>();
            auto rcst = rhs.getDefiningOp<arith::ConstantIntOp>();

            if(lhs == rhs)
            {
                auto zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
                rewriter.replaceOp(op, zero);
                return success();
            }
            else
            {
                return failure();
            }

        }//

    };

    // / 1 pattern
    struct DividOnePattern : OpRewritePattern<arith::DivSIOp>
    {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::DivSIOp op, PatternRewriter &rewriter) const override
        {
            auto rdefOp = op.getRhs().getDefiningOp<arith::ConstantIntOp>();
            if(rdefOp && rdefOp.value() == 1)
            {
                rewriter.replaceOp(op, op.getLhs());
                return success();  
            }

            return failure();

        }//

    };

    struct ConstantMulFoldPattern : OpRewritePattern<arith::MulIOp>
    {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(arith::MulIOp op, PatternRewriter &rewriter) const override
        {
            auto lhs = op.getLhs(), rhs = op.getRhs();
            auto lcst = lhs.getDefiningOp<arith::ConstantIntOp>();
            auto rcst = rhs.getDefiningOp<arith::ConstantIntOp>();

            if(lcst != nullptr && rcst != nullptr)
            {
                llvm::outs() << "Found constant muli\n";
                auto newValue = rewriter.create<arith::ConstantIntOp>(op.getLoc(), lcst.value() * rcst.value(), 32);
                rewriter.replaceOp(op, newValue);
                return success();
            }
            else
            {
                return failure();
            }

        }

    };

    struct ICPatternPass : PassWrapper<ICPatternPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ICPatternPass);

        StringRef getArgument() const override
        {
            return "icpattern";
        }

        StringRef getDescription() const override
        {
            return "icpattern elimination!";
        }

        void runOnOperation() override
        {
            //Init
            RewritePatternSet patterns(&getContext());
            patterns.add<AddZeroPattern>(&getContext());
            patterns.add<MulOnePattern>(&getContext());
            patterns.add<ConstantFoldPattern>(&getContext());
            patterns.add<SubSelfPattern>(&getContext());
            patterns.add<ConstantMulFoldPattern>(&getContext());
            patterns.add<DividOnePattern>(&getContext());

            //greedy traverse
            GreedyRewriteConfig config;
            config.enableFolding(false);

            //perform pattern
            if(failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
            {
                signalPassFailure();
            }//

            //return
            return;

        }//void

    };

    //pass - Linaglg analysis 
    struct LinalgAnalysisPass : PassWrapper<LinalgAnalysisPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgAnalysisPass);

        StringRef getArgument() const override
        {
            return "linalg-analysis";
        }

        StringRef getDescription() const override
        {
            return "print matmul tensor types";
        }

        void runOnOperation() override
        {
            getOperation()->walk([](linalg::LinalgOp op)
            {
                llvm::errs() << "operation: " << op.getOperation()->getName() << "\n";

                //print type
                auto operands = op.getOperation()->getOperands();
                llvm::errs() << "A: " << operands[0].getType() << "\n";
                llvm::errs() << "B: " << operands[1].getType() << "\n";
                llvm::errs() << "C: " << operands[2].getType() << "\n";
                for(Value result : op.getOperation()->getResults())
                {
                    llvm::errs() << "Result: " << result.getType() << "\n";
                }

                //print properties
                auto aType = cast<RankedTensorType>(operands[0].getType());
                auto bType = cast<RankedTensorType>(operands[1].getType());
                int64_t M = aType.getDimSize(0);
                int64_t K = aType.getDimSize(1);
                int64_t N = bType.getDimSize(1);
                llvm::errs() << "M = " << M << ", K = " << K << ", N = " << N << "\n";

                //print index
                auto maps = op.getIndexingMapsArray();
                llvm::errs() << "A map: " << maps[0] << "\n";
                llvm::errs() << "B map: " << maps[1] << "\n";
                llvm::errs() << "C map: " << maps[2] << "\n"; 

                //print iterators
                auto iteratorTypes = op.getIteratorTypesArray();
                for(unsigned index = 0; index < iteratorTypes.size(); index++)
                {
                    llvm::errs() << "LLoop: " << index << ": " << utils::stringifyIteratorType(iteratorTypes[index]) << "\n";
                }
                llvm::outs() << "\n";
                
            });

        }//void
        
    };

    //pass - Fusion
    struct FusionPass : PassWrapper<FusionPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass);

        StringRef getArgument() const override
        {
            return "toy-fusion";
        }

        StringRef getDescription() const override
        {
            return "tensor fusion operations";
        }

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect>();
        }

        void runOnOperation() override
        {
            //pattern set
            RewritePatternSet patterns(&getContext());

            //populateElementwiseOpsFusionPatterns
            linalg::populateElementwiseOpsFusionPatterns(
                patterns, 
                [](OpOperand* operand)
                {
                    Operation* producer = operand->get().getDefiningOp();
                    return producer && producer->hasOneUse();
                });

            //greedy use patterns
            GreedyRewriteConfig config;
            config.enableFolding(false);
            if(failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
            {
                signalPassFailure();
            }

        }//voide

    };// fusion pass

    //pass - Tiling
    struct TilingPass : PassWrapper<TilingPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilingPass);

        StringRef getArgument() const override
        {
            return "toy-tiling";
        }

        StringRef getDescription() const override
        {
            return "tiling pass";
        }

        void getDependentDialects(DialectRegistry &registry) const override
        {
            registry.insert<affine::AffineDialect, arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
            linalg::registerTilingInterfaceExternalModels(registry);
        }

        void runOnOperation() override
        {
            getOperation()->walk([this](func::FuncOp func)
            {
                //only check "add_then_mul"
                if(func.getSymName() != "add_then_mul")
                {
                    return;
                }

                //Init
                IRRewriter rewriter(&getContext());

                //collect all generic ops
                SmallVector<linalg::GenericOp> geOps;
                func.walk([&geOps](linalg::GenericOp op)
                {
                    geOps.push_back(op);
                });

                //tiling
                int64_t m = 2, n = 2;
                for(linalg::GenericOp op : geOps)
                {
                    //set insert position
                    rewriter.setInsertionPoint(op);

                    //set tile dimensions
                    scf::SCFTilingOptions options;
                    options.setTileSizes({rewriter.getIndexAttr(m), rewriter.getIndexAttr(n)});

                    //create tile and its loop
                    auto tiled = scf::tileUsingSCF(rewriter, cast<TilingInterface>(op.getOperation()), options);
                    if(failed(tiled))
                    {
                        op->emitError("tiling failed");
                        signalPassFailure();
                        return;
                    }

                    //replace original tensor computation
                    rewriter.replaceOp(op, tiled->replacements);
                }
                
                
            });

        }//runOnOperation

    };

    //pass - ToyPromoteAccumulatorPass
    struct PaPattern : OpRewritePattern<scf::ForOp>
    {
        using OpRewritePattern::OpRewritePattern;
        struct TargetOps
        {
            memref::LoadOp loadOp1, loadOp2, loadOp3;
            arith::MulFOp mulOp;
            arith::AddFOp addOp;
            memref::StoreOp storeOp;
        }; //

        bool isTargetLoop(scf::ForOp loop) const
        {
            //check validity
            if(loop == nullptr)
            {
                return false;
            }

            //check
            auto func = loop.getOperation()->getParentOfType<func::FuncOp>();
            if(func == nullptr || !func->hasAttr("toy.accumulator_safe") || func.getNumArguments() != 3 || loop.getNumResults() != 0 || !loop->getAttrs().empty())
            {
                return false;
            }

            if(!isConstantIntValue(loop.getLowerBound(), 0) || !isConstantIntValue(loop.getUpperBound(), 64) || !isConstantIntValue(loop.getStep(), 1))
            {
                return false;
            }

            //return
            return true;
        }//

        bool hasTargetOps(scf::ForOp loop) const
        {
            //get necessary parameters
            SmallVector<Value> outsideValue = getOutsideValues(loop);
            SmallVector<Operation*> insideOps = getInsideOps(loop);
            SmallVector<Value> outsideMatrixs = getOutsideMatrixs(loop);
            if(outsideValue.size() != 4)
            {
                return false;
            }

            //running reuslt
            SmallVector<Value> runningResults;

            //check load ops
            if(!checkLoadOps(insideOps, outsideMatrixs, outsideValue[2], outsideValue[3], loop.getInductionVar(), runningResults))
            {
                return false;
            }

            //check mul ops
            if(!checkMulOps(insideOps, runningResults[0], runningResults[1], runningResults))
            {
                return false;
            }

            //check add ops
            if(!checkAddOps(insideOps, runningResults[3], runningResults[2], runningResults))
            {
                return false;
            }

            //check store ops
            if(!checkStoreOps(insideOps, runningResults[4], outsideMatrixs[2], outsideValue[2], outsideValue[3]))
            {
                return false;
            }
            
            //return
            return insideOps.empty();
        }//

        SmallVector<Operation*> getInsideOps(scf::ForOp loop) const
        {
            //check validity
            if(loop == nullptr)
            {
                return {};
            }

            //get all of Ops inside the body
            SmallVector<Operation*> ops;
            for(Operation& op : loop.getBody()->without_terminator())
            {
                ops.push_back(&op);
            }

            //return
            return ops;
        }

        SmallVector<Value> getOutsideValues(scf::ForOp loop) const
        {
            //check validity
            if(loop == nullptr)
            {
                return {};
            }

            //get gpu.lunch body
            auto launch = loop.getOperation()->getParentOfType<gpu::LaunchOp>();
            if(launch == nullptr)
            {
                return {};
            }

            //get 1st block
            SmallVector<Value> results;
            for(Operation& op : launch.getBody().front())
            {
                if(&op == loop.getOperation())
                {
                    break;
                }

                for(Value result : op.getResults())
                {
                    results.push_back(result);
                }

            }//

            //return
            return results;
        }

        SmallVector<Value> getOutsideMatrixs(scf::ForOp loop) const
        {
            //check validity
            if(loop == nullptr)
            {
                return {};
            }

            //get matrixs
            SmallVector<Value> matrixs;
            auto func = loop.getOperation()->getParentOfType<func::FuncOp>();
            for(int i = 0; i < 3; i++)
            {
                matrixs.push_back(func.getArgument(i));
            }
            
            //return
            return matrixs;
        }

        bool checkLoadOps(SmallVector<Operation*>& ops, SmallVector<Value> matrixs, Value i, Value j, Value k, SmallVector<Value>& results) const
        {
            //get all of load ops
            if(ops.empty() || matrixs.size() != 3)
            {
                return false;
            }

            //get all load ops
            SmallVector<memref::LoadOp> loadOps;
            while(true)
            {
                if(ops.empty() || !isa<memref::LoadOp>(ops.front()))
                {
                    break;
                }

                //body
                loadOps.push_back(dyn_cast<memref::LoadOp>(ops.front()));
                ops.erase(ops.begin());
            }

            if(loadOps.size() != 3)
            {
                return false;
            }

             //match two indicies
            for(memref::LoadOp loadOp : loadOps)
            {
                if(loadOp.getIndices().size() != 2)
                {
                    return false;
                }
            }

            //check all load ops
            Value A  = loadOps[0].getMemref(), B = loadOps[1].getMemref(), C = loadOps[2].getMemref();
            Value Ai = loadOps[0].getIndices()[0], Aj = loadOps[0].getIndices()[1];
            Value Bi = loadOps[1].getIndices()[0], Bj = loadOps[1].getIndices()[1];
            Value Ci = loadOps[2].getIndices()[0], Cj = loadOps[2].getIndices()[1];

            //match input parameters
            if(A != matrixs[0] || B != matrixs[1] || C != matrixs[2])
            {
                return false;
            }
            
           

            //match gemm multiplication indicies
            if(Ai != Ci || Aj != k || Bi != k || Bj != Cj || Ci != i || Cj != j)
            {
                return false;
            }
            
            //update retunr value;
            results.push_back(loadOps[0].getResult());
            results.push_back(loadOps[1].getResult());
            results.push_back(loadOps[2].getResult());

            //return
            return true;
        }//

        bool checkMulOps(SmallVector<Operation*>& ops, Value lhs, Value rhs, SmallVector<Value>& results) const
        {
            //check validity
            if(ops.empty())
            {
                return false;
            }

            //get all mulops
            SmallVector<arith::MulFOp> mulOps;
            while(true)
            {
                if(ops.empty() || !isa<arith::MulFOp>(ops.front()))
                {
                    break;
                }

                //body
                mulOps.push_back(dyn_cast<arith::MulFOp>(ops.front()));
                ops.erase(ops.begin());
            }

            if(mulOps.size() != 1)
            {
                return false;
            }

            //check mul ops
            bool flag = false;
            if(mulOps[0].getLhs() == lhs && mulOps[0].getRhs() == rhs)
            {
                flag = true;
                results.push_back(mulOps[0].getResult());
            }

            //return
            return flag;

        }//

        bool checkAddOps(SmallVector<Operation*>& ops, Value lhs, Value rhs, SmallVector<Value>& results) const
        {
            //check validity
            if(ops.empty())
            {
                return false;
            }

            //get all mulops
            SmallVector<arith::AddFOp> addOps;
            while(true)
            {
                if(ops.empty() || !isa<arith::AddFOp>(ops.front()))
                {
                    break;
                }

                //body
                addOps.push_back(dyn_cast<arith::AddFOp>(ops.front()));
                ops.erase(ops.begin());
            }

            if(addOps.size() != 1)
            {
                return false;
            }

            //check add ops
            bool flag = false;
            if(addOps[0].getLhs() == lhs && addOps[0].getRhs() == rhs)
            {
                flag = true;
                results.push_back(addOps[0].getResult());
            }

            //return
            return flag;

        }//

        bool checkStoreOps(SmallVector<Operation*>& ops, Value lhs, Value C, Value i, Value j) const
        {
            //check validity
            if(ops.empty())
            {
                return false;
            }

            //get all mulops
            SmallVector<memref::StoreOp> storeOps;
            while(true)
            {
                if(ops.empty() || !isa<memref::StoreOp>(ops.front()))
                {
                    break;
                }

                //body
                storeOps.push_back(dyn_cast<memref::StoreOp>(ops.front()));
                ops.erase(ops.begin());
            }

            if(storeOps.size() != 1)
            {
                return false;
            }

            //check store op0
            bool flag = false;
            if(storeOps[0].getValue() == lhs && storeOps[0].getMemref() == C && storeOps[0].getIndices()[0] == i && storeOps[0].getIndices()[1] == j)
            {
                flag = true;

            }

            //return
            return flag;

        }//

        std::optional<TargetOps> findMatchedOps(scf::ForOp loop) const
        {
            //check validity
            if(loop == nullptr || !isTargetLoop(loop) || !hasTargetOps(loop))
            {
                return std::nullopt;
            }

            //get all target ops
            struct TargetOps targetOps;
            SmallVector<Operation*> ops = getInsideOps(loop);
            targetOps.loadOp1 = dyn_cast<memref::LoadOp>(ops[0]);
            targetOps.loadOp2 = dyn_cast<memref::LoadOp>(ops[1]);
            targetOps.loadOp3 = dyn_cast<memref::LoadOp>(ops[2]);
            targetOps.mulOp   = dyn_cast<arith::MulFOp>(ops[3]);
            targetOps.addOp   = dyn_cast<arith::AddFOp>(ops[4]);
            targetOps.storeOp = dyn_cast<memref::StoreOp>(ops[5]);
            
            //return
            return targetOps;
        }//

        void rewriteOps(scf::ForOp loop, const TargetOps &targetOps, PatternRewriter &rewriter) const
        {
            //check validity
            if(loop == nullptr)
            {
                return;
            }

            //creating new loop
            /** get old ops */
            auto loadOp1 = targetOps.loadOp1, loadOp2 = targetOps.loadOp2, loadOp3 = targetOps.loadOp3;
            auto mulOp = targetOps.mulOp;
            Value C = loadOp3.getMemref();
            Value i = loadOp3.getIndices()[0], j = loadOp3.getIndices()[1];

            // create a new loop 
            /* loop head */
            rewriter.setInsertionPoint(loop);
            //auto initLoadOp = rewriter.create<memref::LoadOp>(loop.getLoc(), C, ValueRange{i, j});
            auto constantZeroOp = rewriter.create<arith::ConstantOp>(loop.getLoc(), rewriter.getF32FloatAttr(0.0));
            auto forLoopOp = rewriter.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), ValueRange(constantZeroOp.getResult()));
            if(!forLoopOp.getBody()->empty())
            {
                rewriter.eraseOp(forLoopOp.getBody()->getTerminator());
            }
            
            /* loop body */
            rewriter.setInsertionPointToStart(forLoopOp.getBody());
            IRMapping mapping;
            mapping.map(loop.getInductionVar(), forLoopOp.getInductionVar());

            rewriter.clone(*(loadOp1.getOperation()), mapping);
            rewriter.clone(*(loadOp2.getOperation()), mapping);
            auto newMulOp = dyn_cast<arith::MulFOp>(rewriter.clone(*mulOp.getOperation(), mapping));
            auto newAddOp = rewriter.create<arith::AddFOp>(loop.getLoc(), newMulOp.getResult(), forLoopOp.getRegionIterArgs()[0]);
            rewriter.create<scf::YieldOp>(forLoopOp.getLoc(), ValueRange{newAddOp.getResult()});
            
            /** loop outside */
            rewriter.setInsertionPointAfter(forLoopOp);
            rewriter.create<memref::StoreOp>(loop.getLoc(), forLoopOp.getResult(0), C, ValueRange{i, j});

            /** erase the old Ops */
            rewriter.eraseOp(loop);

            //return
            return;
        }

        struct FillOps
        {
            gpu::LaunchOp launchOp;
            affine::AffineApplyOp rowBaseOp;
            affine::AffineApplyOp colBaseOp;
            arith::AddIOp rowOp;
            arith::AddIOp colOp;
            memref::StoreOp storeOp;
        };

        struct FillOps getFillOps(gpu::LaunchOp fillLaunch) const
        {
            //check
            if (!fillLaunch)
            {
                return {};
            }

            //init
            struct FillOps fillOps{};

            //get
            SmallVector<Operation*> ops;
            for(Operation& op : fillLaunch.getBody().front().without_terminator())
            {
                ops.push_back(&op);
            }

            if(ops.size() == 5)
            {
                fillOps.launchOp = fillLaunch;
                fillOps.rowBaseOp = dyn_cast<affine::AffineApplyOp>(ops[0]);
                fillOps.colBaseOp = dyn_cast<affine::AffineApplyOp>(ops[1]);
                fillOps.rowOp = dyn_cast<arith::AddIOp>(ops[2]);
                fillOps.colOp = dyn_cast<arith::AddIOp>(ops[3]);
                fillOps.storeOp = dyn_cast<memref::StoreOp>(ops[4]);
            }

            //update
            if(!fillOps.launchOp || !fillOps.rowBaseOp || !fillOps.colBaseOp || !fillOps.rowOp || !fillOps.colOp || !fillOps.storeOp)
            {
                fillOps = {};
            }

            //return
            return fillOps;

        }//

        bool isMatchedFillOps(struct FillOps fillOps, Value C) const
        {
            //check
            if(!fillOps.launchOp || !fillOps.rowBaseOp || !fillOps.colBaseOp || !fillOps.rowOp || !fillOps.colOp || !fillOps.storeOp)
            {
                return false;
            }

            if(fillOps.rowBaseOp.getNumOperands() != 1 || fillOps.colBaseOp.getNumOperands() != 1 || fillOps.rowOp.getNumOperands() != 2 || fillOps.colOp.getNumOperands() != 2 || fillOps.storeOp.getIndices().size() != 2 )
            {
                return false;
            }

            //init
            bool flag;
            auto expectedFormula = AffineMap::get(0, 1, getAffineSymbolExpr(0, fillOps.launchOp.getContext()) * 8);
            auto gridSize = fillOps.launchOp.getGridSizeOperandValues();
            auto blockSize = fillOps.launchOp.getBlockSizeOperandValues();
            auto blockId = fillOps.launchOp.getBlockIds();
            auto threadId= fillOps.launchOp.getThreadIds();
            auto rowBaseFormula = fillOps.rowBaseOp.getAffineMap(), colBaseFormula = fillOps.colBaseOp.getAffineMap();
            Value rowBase = fillOps.rowBaseOp.getResult();
            Value colBase = fillOps.colBaseOp.getResult();
            auto rowBaseOpLhs = fillOps.rowBaseOp.getOperand(0);
            auto colBaseOpLhs = fillOps.colBaseOp.getOperand(0);
            Value row = fillOps.rowOp.getResult();
            Value col = fillOps.colOp.getResult();
            auto rowOpLhs = fillOps.rowOp.getOperand(0), rowOpRhs = fillOps.rowOp.getOperand(1); 
            auto colOpLhs = fillOps.colOp.getOperand(0), colOpRhs = fillOps.colOp.getOperand(1);
            Value storeOpLhs = fillOps.storeOp.getValue(), storeOpBuff = fillOps.storeOp.getMemref(), buffI = fillOps.storeOp.getIndices()[0], buffJ = fillOps.storeOp.getIndices()[1];

            //compute
            /** launch input */
            bool flag1 = false;
            if(isConstantIntValue(gridSize.x, 8) && isConstantIntValue(gridSize.y, 8) && isConstantIntValue(gridSize.z, 1) && isConstantIntValue(blockSize.x, 8) && isConstantIntValue(blockSize.y, 8) && isConstantIntValue(blockSize.z, 1))
            {
                flag1 = true;
            }

            /** affine ops */
            bool flag2 = false;
            if(rowBaseFormula == expectedFormula && colBaseFormula == expectedFormula && rowBaseOpLhs == blockId.x && colBaseOpLhs == blockId.y)
            {
                flag2 = true;
            }

            /** add ops */
            bool flag3 = false;
            if(rowOpLhs == threadId.x && rowOpRhs == rowBase && colOpLhs == threadId.y && colOpRhs == colBase)
            {
                flag3 = true;    
            }

            /** store op */
            bool flag4 = false;
            if(matchPattern(storeOpLhs, m_AnyZeroFloat()) && storeOpBuff == C && buffI == row && buffJ == col )
            {
                flag4 = true;
            }

            //update flag
            flag = flag1 && flag2 && flag3 && flag4;

            //return
            return flag;
        }

        bool removeZeroFillLaunch(gpu::LaunchOp fillLaunch, PatternRewriter &rewriter) const
        {
            //get fill ops
            struct FillOps fillOps = getFillOps(fillLaunch);
            if(!fillOps.launchOp && !fillOps.rowBaseOp && !fillOps.colBaseOp && !fillOps.rowOp && !fillOps.colOp && !fillOps.storeOp)
            {
                return false;
            }

            //check
            auto funcOp = fillLaunch.getOperation()->getParentOfType<func::FuncOp>();
            Value bufferC = funcOp.getArgument(2);
            if(!isMatchedFillOps(fillOps, bufferC))
            {
                return false;
            }

            //rewrite
            rewriter.eraseOp(fillLaunch);

            //return
            return true;
        }

        LogicalResult addAccumulator(scf::ForOp loop, PatternRewriter &rewriter) const
        {
            //find mathced ops
            auto targetOps = findMatchedOps(loop);
            if(!targetOps.has_value())
            {
                return failure();
            }

            //rewrite
            rewriteOps(loop, *targetOps, rewriter);

            //return
            return success();
        }
         
        LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter &rewriter) const override
        {
            //check validity
            if(loop == nullptr)
            {
                return failure();
            }

            //init
            LogicalResult flag = failure();
            gpu::LaunchOp curLaunch = loop.getOperation()->getParentOfType<gpu::LaunchOp>();
            gpu::LaunchOp fillLaunch = curLaunch ? dyn_cast_or_null<gpu::LaunchOp>(curLaunch->getPrevNode()) : gpu::LaunchOp{};

            //check
            //check gemm validation
            auto targetGEMMOps = findMatchedOps(loop);
            if(!targetGEMMOps)
            {
                return failure();
            }
            
            //do optimiztion
            if(fillLaunch)
            {
                /** Optimization 1: removing zero filling */
                if(!removeZeroFillLaunch(fillLaunch, rewriter))
                {
                    flag = failure();
                    return flag;
                }

                /** Optimizaiton 2: adding built-in accumulator */
                flag = addAccumulator(loop, rewriter);

            }

            //return
            return flag;
        }//

    };//
    
    struct ToyPromoteAccumulatorPass : PassWrapper<ToyPromoteAccumulatorPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyPromoteAccumulatorPass);
        StringRef getArgument() const override
        {
            return "toy-pa";
        }

        StringRef getDescription() const override
        {
            return "pa pass";
        }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        void runOnOperation() override
        {
            //init
            RewritePatternSet patterns(&getContext());
            patterns.add<PaPattern>(&getContext());
            FrozenRewritePatternSet frozen(std::move(patterns));

            //apply patterns
            if(failed(applyPatternsGreedily(getOperation(), frozen)))
            {
                signalPassFailure();
            }

        }//void

    }; // struct

}//namespace

namespace toy_shared_tile
{   
    //pattern
    struct SharedTiledPattern : OpRewritePattern<gpu::LaunchOp>
    {
        using OpRewritePattern::OpRewritePattern;
        struct GEMMOps
        {
            gpu::LaunchOp launchOp;
            
            affine::AffineApplyOp rowBaseOp;
            affine::AffineApplyOp colBaseOp;
            arith::AddIOp rowOp;
            arith::AddIOp colOp;

            scf::ForOp loopOp;
            memref::LoadOp loadAOp;
            memref::LoadOp loadBOp;
            arith::MulFOp mulOp;
            arith::AddFOp addOp;

            memref::StoreOp storeOp;
        };

        struct GEMMOps collectGEMMOps(gpu::LaunchOp launch) const
        {
            //check validdy
            if(!launch)
            {
                return GEMMOps{};
            }

            //init
            struct GEMMOps gemmOps;

            //collect ps
            SmallVector<Operation*> ops;
            for(Operation& op : launch.getBody().front().without_terminator())
            {
                ops.push_back(&op);
                auto loopOp = dyn_cast<scf::ForOp>(&op);
                if(loopOp)
                {
                    for(Operation& innerOp : loopOp.getBody()->without_terminator())
                    {
                        ops.push_back(&innerOp);
                    }//

                }

            }//for

            if(ops.size() != 10)
            {
                return GEMMOps{};
            }

            //update gemmOps
            gemmOps.launchOp  = launch;
            gemmOps.rowBaseOp = dyn_cast<affine::AffineApplyOp>(ops[0]);
            gemmOps.colBaseOp = dyn_cast<affine::AffineApplyOp>(ops[1]);
            gemmOps.rowOp     = dyn_cast<arith::AddIOp>(ops[2]);
            gemmOps.colOp     = dyn_cast<arith::AddIOp>(ops[3]);
            gemmOps.loopOp    = dyn_cast<scf::ForOp>(ops[4]);
            gemmOps.loadAOp   = dyn_cast<memref::LoadOp>(ops[5]);
            gemmOps.loadBOp   = dyn_cast<memref::LoadOp>(ops[6]);
            gemmOps.mulOp     = dyn_cast<arith::MulFOp>(ops[7]);
            gemmOps.addOp     = dyn_cast<arith::AddFOp>(ops[8]);
            gemmOps.storeOp   = dyn_cast<memref::StoreOp>(ops[9]);

            //validate gemmOps
            if(!gemmOps.launchOp  || 
               !gemmOps.rowBaseOp || !gemmOps.colBaseOp || 
               !gemmOps.rowOp     || !gemmOps.colOp || 
               !gemmOps.loopOp    || 
               !gemmOps.loadAOp   || !gemmOps.loadBOp ||
               !gemmOps.mulOp     || !gemmOps.addOp ||
               !gemmOps.storeOp)
            {
                return GEMMOps{};
            }
        
            //return
            return gemmOps;        
        }

        bool isMatchedGEMMOps(struct GEMMOps gemmOps) const
        {
            //validity
            if(!gemmOps.launchOp)
            {
                return false;
            }
            
            //init
            bool flag = true;
            auto expectedFormula = AffineMap::get(0, 1, getAffineSymbolExpr(0, gemmOps.launchOp.getContext()) * 8);
            auto expectedType = MemRefType::get({64, 64}, Float32Type::get(gemmOps.launchOp.getContext()));

            /** funcOp */
            auto funcOp = gemmOps.launchOp.getOperation()->getParentOfType<func::FuncOp>();
            if(!funcOp || funcOp.getNumArguments() != 3)
            {
                flag = false;
                return flag;
            }

            Value A = funcOp.getArgument(0);
            Value B = funcOp.getArgument(1);
            Value C = funcOp.getArgument(2);
            if(A.getType() != expectedType || B.getType() != expectedType || C.getType() != expectedType)
            {
                flag = false;
                return flag;
            }

            if(!funcOp->hasAttr("toy.accumulator_safe"))
            {
                flag = false;
                return flag;
            }

            /** launchOp */
            auto launchOp   = gemmOps.launchOp;
            auto blocksId   = gemmOps.launchOp.getBlockIds();
            auto threadsId  = gemmOps.launchOp.getThreadIds();
            auto gridSize   = gemmOps.launchOp.getGridSizeOperandValues();
            auto blockSize  = gemmOps.launchOp.getBlockSizeOperandValues();
            auto rowBaseOp  = gemmOps.rowBaseOp;
            auto colBaseOp  = gemmOps.colBaseOp;
            auto rowOp      = gemmOps.rowOp;     
            auto colOp      = gemmOps.colOp;     
            auto loopOp     = gemmOps.loopOp;    
            auto loadAOp    = gemmOps.loadAOp;   
            auto loadBOp    = gemmOps.loadBOp;  
            auto mulOp      = gemmOps.mulOp;     
            auto addOp      = gemmOps.addOp;     
            auto storeOp    = gemmOps.storeOp;   

            //mathch
            /** grid size and block size checking */
            if(!isConstantIntValue(gridSize.x, 8)  || !isConstantIntValue(gridSize.y, 8)  || !isConstantIntValue(gridSize.z, 1) ||
               !isConstantIntValue(blockSize.x, 8) || !isConstantIntValue(blockSize.y, 8) || !isConstantIntValue(blockSize.z, 1))
            {
                flag = false;
                return flag;    
            }
            
            /** rowbase and colbase checking */
            if(rowBaseOp.getAffineMap() != expectedFormula || rowBaseOp.getNumOperands() != 1 || rowBaseOp.getOperand(0) != blocksId.x)
            {
                flag = false;
                return flag;
            }

            if(colBaseOp.getAffineMap() != expectedFormula || colBaseOp.getNumOperands() != 1 || colBaseOp.getOperand(0) != blocksId.y)
            {
                flag = false;
                return flag;
            }

            /** row and col checking */
            if(rowOp.getOperand(0) != threadsId.y  || rowOp.getOperand(1) != rowBaseOp.getResult() || rowOp.getNumOperands() != 2)
            {
                flag = false;
                return flag;
            }

            if(colOp.getOperand(0) != threadsId.x  || colOp.getOperand(1) != colBaseOp.getResult() || colOp.getNumOperands() != 2)
            {
                flag = false;
                return flag;
            }

            /** for loop head checking */
            if(loopOp.getInitArgs().size() != 1 || loadAOp.getIndices().size() != 2 || loadBOp.getIndices().size() != 2 || mulOp.getNumOperands() != 2 || addOp.getNumOperands() != 2 || storeOp.getNumOperands() != 4 || 
               storeOp.getIndices().size() != 2)
            {
                flag = false;
                return flag;
            }

            Value k = loopOp.getInductionVar();
            Value upper = loopOp.getUpperBound();
            Value lower = loopOp.getLowerBound();
            Value step = loopOp.getStep();
            auto initArgs = loopOp.getInitArgs();
            auto curSum = loopOp.getRegionIterArgs()[0];
            if(!matchPattern(initArgs[0], m_AnyZeroFloat()))
            {
                flag = false;
                return flag;
            }

            if(!isConstantIntValue(lower, 0) || !isConstantIntValue(upper, 64) || !isConstantIntValue(step, 1))
            {
                flag = false;
                return flag;
            }

            /** inside Ops checking */
            Value loadABuff = loadAOp.getMemref(), aI = loadAOp.getIndices()[0], aJ = loadAOp.getIndices()[1], loadAResult = loadAOp.getResult();
            Value loadBBuff = loadBOp.getMemref(), bI = loadBOp.getIndices()[0], bJ = loadBOp.getIndices()[1], loadBResult = loadBOp.getResult();
            if(loadABuff != A || aI != rowOp.getResult() || aJ != k)
            {
                flag = false;
                return flag;
            }

            if(loadBBuff != B || bI != k || bJ != colOp.getResult())
            {
                flag = false;
                return flag;
            }

            Value mulLhs = mulOp.getLhs(), mulRhs = mulOp.getRhs(), mulResult = mulOp.getResult();
            Value addLhs = addOp.getLhs(), addRhs = addOp.getRhs(), addResult = addOp.getResult();
            if(mulLhs != loadAResult || mulRhs != loadBResult)
            {
                flag = false;
                return flag;
            }

            if(addLhs != mulResult || addRhs != curSum)
            {
                flag = false;
                return flag;
            }

            auto yieldOp = dyn_cast<scf::YieldOp>(loopOp.getBody()->getTerminator());
            if(!yieldOp || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != addResult)
            {
                flag = false;
                return flag;
            }

            Value storeOpLhs = storeOp.getValue(), storeOpBuff = storeOp.getMemref(), cI = storeOp.getIndices()[0], cJ = storeOp.getIndices()[1];
            if(storeOpLhs != loopOp.getResult(0) || storeOpBuff != C || cI != rowOp.getResult() || cJ != colOp.getResult())
            {
                flag = false;
                return flag;
            }

            //return
            return flag;
        }

        void rewriteGEMM(GEMMOps gemmOps, PatternRewriter& rewriter) const
        {
            //get all required ops
            auto funcOp = gemmOps.launchOp.getOperation()->getParentOfType<func::FuncOp>();
            Value A = funcOp.getArgument(0);
            Value B = funcOp.getArgument(1);
            Value C = funcOp.getArgument(2);

            auto launchOp   = gemmOps.launchOp;
            auto loc        = launchOp.getLoc();
            auto blocksId   = gemmOps.launchOp.getBlockIds();
            auto threadsId  = gemmOps.launchOp.getThreadIds();
            auto gridSize   = gemmOps.launchOp.getGridSizeOperandValues();
            auto blockSize  = gemmOps.launchOp.getBlockSizeOperandValues();
            auto rowBaseOp  = gemmOps.rowBaseOp;
            auto colBaseOp  = gemmOps.colBaseOp;
            auto rowOp      = gemmOps.rowOp;     
            auto colOp      = gemmOps.colOp;     
            auto loopOp     = gemmOps.loopOp;    
            auto loadAOp    = gemmOps.loadAOp;   
            auto loadBOp    = gemmOps.loadBOp;  
            auto mulOp      = gemmOps.mulOp;     
            auto addOp      = gemmOps.addOp;     
            auto storeOp    = gemmOps.storeOp;   

            //create shared memory
            auto sharedSpace = gpu::AddressSpaceAttr::get(rewriter.getContext(), gpu::AddressSpace::Workgroup);
            auto sharedType = MemRefType::get({8, 8}, rewriter.getF32Type(), MemRefLayoutAttrInterface{}, sharedSpace);
            Value As, Bs;
            rewriter.modifyOpInPlace(launchOp, [&]{ As = launchOp.addWorkgroupAttribution(sharedType, loc); Bs = launchOp.addWorkgroupAttribution(sharedType, loc); } );

            //rewrite
            /** create outer loop head  */
            rewriter.setInsertionPoint(loopOp);
            Value lower = loopOp.getLowerBound();
            Value upper = loopOp.getUpperBound();
            Value tileSize = blockSize.x;
            Value initialSum = loopOp.getInitArgs()[0];
            auto outerLoop = rewriter.create<scf::ForOp>(loc, lower, upper, tileSize, ValueRange{initialSum});

            /** create outer loop body */
            rewriter.setInsertionPointToStart(outerLoop.getBody());
            Value k   = outerLoop.getInductionVar();
            Value tx  = threadsId.x, ty = threadsId.y;
            Value row = rowOp.getResult(), col = colOp.getResult();
            Value ak  = rewriter.create<arith::AddIOp>(loc, k, tx).getResult();
            Value bk  = rewriter.create<arith::AddIOp>(loc, k, ty).getResult();
            Value a   = rewriter.create<memref::LoadOp>(loc, A, ValueRange{row, ak}).getResult();
            Value b   = rewriter.create<memref::LoadOp>(loc, B, ValueRange{bk, col}).getResult();
            rewriter.create<memref::StoreOp>(loc, a, As, ValueRange{ty, tx});
            rewriter.create<memref::StoreOp>(loc, b, Bs, ValueRange{ty, tx});
            rewriter.create<gpu::BarrierOp>(loc);

            /** create inner loop head*/
            Value llower   = loopOp.getLowerBound();
            Value uupper   = blockSize.x;
            Value stepSize = blockSize.z;
            auto innerLoop = rewriter.create<scf::ForOp>(loc, llower, uupper, stepSize, ValueRange{outerLoop.getRegionIterArgs()[0]});
            
            /** compute innerLoop sum */ 
            rewriter.setInsertionPointToStart(innerLoop.getBody());
            Value kk = innerLoop.getInductionVar();
            Value prevInnerSum = innerLoop.getRegionIterArgs()[0];
            Value aa = rewriter.create<memref::LoadOp>(loc, As, ValueRange{ty, kk}).getResult();
            Value bb = rewriter.create<memref::LoadOp>(loc, Bs, ValueRange{kk, tx}).getResult();
            Value mulAB  = rewriter.create<arith::MulFOp>(loc, aa, bb).getResult();
            Value curInnerSum = rewriter.create<arith::AddFOp>(loc, mulAB, prevInnerSum).getResult();
            rewriter.create<scf::YieldOp>(loc, curInnerSum);

            /** computer outerLoop sum */
            rewriter.setInsertionPointAfter(innerLoop);
            rewriter.create<gpu::BarrierOp>(loc);
            rewriter.create<scf::YieldOp>(loc, innerLoop.getResults());

            /** update C[row][col] */
            rewriter.setInsertionPointAfter(outerLoop);
            rewriter.create<memref::StoreOp>(loc, outerLoop.getResult(0), C, ValueRange{row, col});
            rewriter.eraseOp(storeOp);

            /** remove old loop */
            rewriter.replaceOp(loopOp, outerLoop.getResults());
            
            //return
            return;
        }

        LogicalResult matchAndRewrite(gpu::LaunchOp launch, PatternRewriter &rewriter) const override
        {
            //check validity
            if(!launch)
            {
                return failure();
            }

            //collect gemmOps
            struct GEMMOps gemmOps = collectGEMMOps(launch);
            if(!gemmOps.launchOp)
            {
                return failure();
            }

            //validate gemmOps
            if(!isMatchedGEMMOps(gemmOps))
            {
                return failure();
            }

            //tiling the gemm
            rewriteGEMM(gemmOps, rewriter);
            
            //return
            return success();
        }//

    };//

    //pass
    struct ToySharedTilePass : PassWrapper<ToySharedTilePass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToySharedTilePass);
        StringRef getArgument() const override
        {
            return "toy-shared-tile";
        }

        StringRef getDescription() const override
        {
            return "Tile the fixed thread-swapped GEMM with 8x8 shared-memory tiles";
        }

        void runOnOperation() override
        {
            //init
            RewritePatternSet patterns(&getContext());
            patterns.add<SharedTiledPattern>(&getContext());
            FrozenRewritePatternSet frozen(std::move(patterns));

            //apply
            walkAndApplyPatterns(getOperation(), frozen);

        }//void

    }; // struct

}//namespace

namespace toy_mlp_fusion
{
    //struct 
    struct GBROps
    {
        linalg::MatmulOp gemmOp;
        linalg::GenericOp biasOp;
        linalg::GenericOp reluOp;
    };

    //pattern
    struct ToyGEMMBiasReluFusionPattern : OpRewritePattern<linalg::GenericOp>
    {
        using OpRewritePattern::OpRewritePattern;

        struct GBROps collectOps(linalg::GenericOp op) const
        {
            //validity
            if(!op)
            {
                return GBROps{};
            }

            //init
            struct GBROps gbrOps{};

            //get ReluOp
            auto biasOp = dyn_cast_or_null<linalg::GenericOp>(op->getPrevNode());
            if(!biasOp)
            {
                return {};
            }

            auto gemmOp = dyn_cast_or_null<linalg::MatmulOp>(biasOp->getPrevNode());
            if(!gemmOp)
            {
                return {};
            }
            
            //update gbr
            gbrOps.gemmOp = gemmOp;
            gbrOps.biasOp = biasOp;
            gbrOps.reluOp = op;
        
            //return
            return gbrOps;
        }
        
        bool checkGEMM(GBROps ops) const
        {
            //init
            bool flag = true;

            //checking
            auto gemmOp = ops.gemmOp;
            if(gemmOp.getNumOperands() != 3)
            {
                flag = false;
                return flag;
            }

            Value X  = gemmOp.getOperation()->getOperand(0);
            Value W1 = gemmOp.getOperation()->getOperand(1);
            Value T1 = gemmOp.getOperation()->getOperand(2);
            auto expectedType = MemRefType::get({64, 64}, Float32Type::get(gemmOp.getContext()));
            if(X.getType() != expectedType || W1.getType() != expectedType || T1.getType() != expectedType)
            {
                flag = false;
                return flag;
            }

            //return
            return flag;
        }

        bool checkBias(GBROps ops, Value T1) const
        {
            //init
            auto flag = true;

            //checking
            /** head checking */
            auto biasOp = ops.biasOp;
            if(!biasOp)
            {
                flag = false;
                return flag;
            }

            auto maps = biasOp.getIndexingMapsArray();
            auto matrixMap = AffineMap::getMultiDimIdentityMap(2, biasOp.getContext());
            auto biasMap   = AffineMap::get(2, 0, getAffineDimExpr(1, biasOp.getContext()));
            if(maps.size() != 3 || maps[0] != matrixMap || maps[1] != biasMap || maps[2] != matrixMap)
            {
                flag = false;
                return flag;
            }

            auto iterators = biasOp.getIteratorTypesArray();
            if(iterators.size() != 2 || iterators[0] != utils::IteratorType::parallel || iterators[1] != utils::IteratorType::parallel)
            {
                flag = false;
                return flag;
            }

            if(biasOp.getNumDpsInputs() != 2 || biasOp.getNumDpsInits() != 1)
            {
                flag = false;
                return flag;
            }
            
            Value biasInput = biasOp.getDpsInputs()[0];
            Value B1 = biasOp.getDpsInputs()[1];
            Value T2 = biasOp.getDpsInits()[0];
            if(biasInput != T1)
            {
                flag = false;
                return flag;
            }

            auto matrixType = MemRefType::get({64, 64}, Float32Type::get(biasOp.getContext()));
            auto biasType = MemRefType::get({64}, Float32Type::get(biasOp.getContext()));
            if(biasInput.getType() != matrixType || B1.getType() != biasType || T2.getType() != matrixType)
            {
                flag = false;
                return flag;
            }

            /** body checking */
            Block& body = biasOp.getOperation()->getRegion(0).front();
            SmallVector<Operation*> bodyOps;
            if(body.getNumArguments() != 3)
            {
                flag = false;
                return flag;
            }

            for(Operation& op : body)
            {
                bodyOps.push_back(&op);
            }

            if(bodyOps.size() != 2)
            {
                flag = false;
                return flag;
            }

            auto addOp = dyn_cast<arith::AddFOp>(bodyOps[0]);
            auto yieldOp = dyn_cast<linalg::YieldOp>(bodyOps[1]);
            auto f32Type = Float32Type::get(biasOp.getContext());
            if(body.getArgument(0).getType() != f32Type || body.getArgument(1).getType() != f32Type || body.getArgument(2).getType() != f32Type ||
                !addOp || !yieldOp || yieldOp.getOperation()->getNumOperands() != 1 ||
                addOp.getResult().getType() != f32Type || yieldOp.getOperand(0).getType() != f32Type)
            {
                flag = false;
                return flag;
            }

            Value w = body.getArgument(0);
            Value b = body.getArgument(1);
            if(addOp.getOperand(0) != w || addOp.getOperand(1) != b || yieldOp.getOperand(0) != addOp.getResult())
            {
                flag = false;
                return flag;
            }

            //return
            return flag;
        }

        bool checkRelu(GBROps ops, Value T2) const
        {
            //init
            bool flag = true;

            //checking
            /** head checking */
            auto reluOp = ops.reluOp;
            if(!reluOp)
            {
                flag = false;
                return flag;
            }

            auto maps = reluOp.getIndexingMapsArray();
            auto matrixMap = AffineMap::getMultiDimIdentityMap(2, reluOp.getContext());
            if(maps.size() != 2 || maps[0] != matrixMap || maps[1] != matrixMap)
            {
                flag = false;
                return flag;
            }

            auto iterators = reluOp.getIteratorTypesArray();
            if(iterators.size() != 2 || iterators[0] != utils::IteratorType::parallel || iterators[1] != utils::IteratorType::parallel)
            {
                flag = false;
                return flag;
            }

            if(reluOp.getNumDpsInputs() != 1 || reluOp.getNumDpsInits() != 1)
            {
                flag = false;
                return flag;
            }

            auto reluInput = reluOp.getDpsInputs()[0];
            auto reluOutput = reluOp.getDpsInits()[0];
            if(!reluInput || reluInput != T2 || !reluOutput )
            {
                flag = false;
                return flag;
            }

            auto matrixType = MemRefType::get({64, 64}, Float32Type::get(reluOp.getContext()));
            if(reluInput.getType() != matrixType || reluOutput.getType() != matrixType)
            {
                flag = false;
                return flag;
            }

            /** body checking */
            Block& body = reluOp.getOperation()->getRegion(0).front();
            SmallVector<Operation*> bodyOps;
            if(body.getNumArguments() != 2)
            {
                flag = false;
                return flag;
            }

            for(Operation& op : body)
            {
                bodyOps.push_back(&op);
            }

            if(bodyOps.size() != 2)
            {
                flag = false;
                return flag;
            }

            auto maximumOp = dyn_cast<arith::MaximumFOp>(bodyOps[0]);
            auto yieldOp = dyn_cast<linalg::YieldOp>(bodyOps[1]);
            auto f32Type = Float32Type::get(reluOp.getContext());
            if(body.getArgument(0).getType() != f32Type || body.getArgument(1).getType() != f32Type ||
                !maximumOp || !yieldOp || yieldOp.getOperation()->getNumOperands() != 1 ||
                maximumOp.getResult().getType() != f32Type || yieldOp.getOperand(0).getType() != f32Type)
            {
                flag = false;
                return flag;
            }

            Value w = body.getArgument(0);
            if(maximumOp.getOperand(0) != w || !matchPattern(maximumOp.getOperand(1), m_AnyZeroFloat()) || yieldOp.getOperand(0) != maximumOp.getResult())
            {
                flag = false;
                return flag;
            }
            
            //return
            return flag;
        }

        bool checkT1Buffer(Value T1, GBROps ops) const
        {
            //check
            SmallVector<Operation*> usersT1;
            auto deallocT1Op = memref::findDealloc(T1);
            if(!deallocT1Op || !(*deallocT1Op) || !isa<memref::DeallocOp>(*deallocT1Op))
            {
                return false;
            }
                
            usersT1.push_back(*deallocT1Op);
            usersT1.push_back(ops.gemmOp);
            usersT1.push_back(ops.biasOp);
            auto fillOp = dyn_cast_or_null<linalg::FillOp>(ops.gemmOp->getPrevNode());
            if(!fillOp || fillOp.getDpsInits()[0] != T1)
            {
                return false;
            }

            usersT1.push_back(fillOp);
            
            auto actualUsersT1 = T1.getUsers();
            if(!std::is_permutation(usersT1.begin(), usersT1.end(), actualUsersT1.begin(), actualUsersT1.end()))
            {
                return false;
            }

            //return
            return true;
        }

        bool checkT2Buffer(Value T2, GBROps ops) const
        {
            //check
            SmallVector<Operation*> usersT2;
            auto deallocT2Op = memref::findDealloc(T2);
            if(!deallocT2Op || !(*deallocT2Op) || !isa<memref::DeallocOp>(*deallocT2Op))
            {
                return false;
            }

            usersT2.push_back(*deallocT2Op);
            usersT2.push_back(ops.biasOp);
            usersT2.push_back(ops.reluOp);
            
            auto actualUsersT2 = T2.getUsers();
            if(!std::is_permutation(usersT2.begin(), usersT2.end(), actualUsersT2.begin(), actualUsersT2.end()))
            {
                return false;
            }

            //return
            return true;
        }

        bool checkIntermediateBuffers(GBROps ops) const
        {
            //init
            Value T1 = ops.gemmOp.getDpsInits()[0];
            Value T2 = ops.biasOp.getDpsInits()[0];
            if(!T1.getDefiningOp<memref::AllocOp>() || !T2.getDefiningOp<memref::AllocOp>())
            {
                return false;
            }
            
            //check
            bool flag = checkT1Buffer(T1, ops) && checkT2Buffer(T2, ops);

            //return
            return flag;
        }
        
        bool checkOutputBuffers(GBROps ops) const
        {
            //init
            auto reluOp = ops.reluOp;

            //check if alocation op
            Value H = ops.reluOp.getDpsInits()[0];
            if(!H.getDefiningOp<memref::AllocOp>())
            {
                return false;
            }

            //check if there are valid users
            bool flag = true;
            for(Operation* user : H.getUsers())
            {
                if(user != reluOp.getOperation())
                {
                    if(user->getBlock() != reluOp->getBlock() || !reluOp->isBeforeInBlock(user))
                    {
                        flag = false;
                        break;
                    }
                }

            }

            if(!flag)
            {
                return false;
            }
            
            //return
            return true;
        }

        bool isMatchedOps(GBROps ops) const
        {
            //validity
            if(!ops.gemmOp)
            {
                return false;
            }

            //init
            auto gemmOp = ops.gemmOp;
            auto biasOp = ops.biasOp;
            auto reluOp = ops.reluOp;
            if(!gemmOp || !biasOp || !reluOp)
            {
                return false;
            }
            
            //checking
            /** gemmOp checking */
            bool isMatchedGEMMOp = checkGEMM(ops);
            if(!isMatchedGEMMOp)
            {
                return false;
            }           

            /** biasOp checking */
            bool isMatchedBiasOp = checkBias(ops, ops.gemmOp.getOperation()->getOperand(2));
            if(!isMatchedBiasOp)
            {
                return false;
            }

            /** reluOp checking */
            bool isMatchedReluOp = checkRelu(ops, ops.biasOp.getDpsInits()[0]);
            if(!isMatchedReluOp)
            {
                return false;
            }

            /** intermediate buffer checking */
            bool isValidIntermediateBuffers = checkIntermediateBuffers(ops);
            if(!isValidIntermediateBuffers)
            {
                return false;
            }

            /** output buffer checking */
            bool isValidOutputBuffers = checkOutputBuffers(ops);
            if(!isValidOutputBuffers)
            {
                return false;
            }

            //return
            return true;
        }

        Value cloneBody(Block& body, ValueRange arguments, PatternRewriter& rewriter) const
        {
            //init
            IRMapping mapping;
            
            //clone
            /** replacing head */
            for(unsigned i = 0; i < arguments.size(); i++)
            {
                mapping.map(body.getArgument(i), arguments[i]);
            }

            /** clone ops */
            for(Operation& op : body.without_terminator())
            {
                rewriter.clone(op, mapping);
            }

            //update return value
            auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
            Value result = mapping.lookupOrDefault(yieldOp.getOperand(0));
            
            //return
            return result;
        }

        bool rewriteOps(GBROps ops, PatternRewriter& rewriter) const
        {
            //validity
            if(!ops.gemmOp)
            {
                return false;
            }

            //init
            bool flag = true;
            
            auto gemmOp = ops.gemmOp;
            auto biasOp = ops.biasOp;
            auto reluOp = ops.reluOp;
            Value X  = gemmOp.getDpsInputs()[0];
            Value W1 = gemmOp.getDpsInputs()[1];
            Value B1 = biasOp.getDpsInputs()[1];
            Value Y  = reluOp.getDpsInits()[0];

            //rewrite
            /* constants for loops' head */
            Location loc = gemmOp.getLoc();
            rewriter.setInsertionPoint(gemmOp);
            Value c0  = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            Value c64 = rewriter.create<arith::ConstantIndexOp>(loc, 64);
            Value c1  = rewriter.create<arith::ConstantIndexOp>(loc, 1);

            /* row and column loop head */
            auto iLoop = rewriter.create<scf::ForOp>(loc, c0, c64, c1);
            rewriter.setInsertionPointToStart(iLoop.getBody());
            auto jLoop = rewriter.create<scf::ForOp>(loc, c0, c64, c1);
            rewriter.setInsertionPointToStart(jLoop.getBody());
            
            /** row and colum loop body */
            Value i = iLoop.getInductionVar();
            Value j = jLoop.getInductionVar();
            Value zero = rewriter.create<arith::ConstantFloatOp>(loc, rewriter.getF32Type(), APFloat(0.0f));

            /** k loop */
            auto kLoop = rewriter.create<scf::ForOp>(loc, c0, c64, c1, ValueRange{zero});

            /** compute X[i, k] * W1[k, j] */
            rewriter.setInsertionPointToStart(kLoop.getBody());
            Value k = kLoop.getInductionVar();
            Value a = rewriter.create<memref::LoadOp>(loc, X, ValueRange{i, k});
            Value b = rewriter.create<memref::LoadOp>(loc, W1, ValueRange{k, j});
            Value acc = kLoop.getRegionIterArgs()[0];
            Value sum = cloneBody(gemmOp.getOperation()->getRegion(0).front(), {a, b, acc}, rewriter);
            rewriter.create<scf::YieldOp>(loc, sum);

            /** compute bias and relu */
            rewriter.setInsertionPointAfter(kLoop);
            Value gemmValue = kLoop.getResult(0);
            Value bias = rewriter.create<memref::LoadOp>(loc, B1, ValueRange{j});
            Value biasSum   = cloneBody(biasOp.getOperation()->getRegion(0).front(), {gemmValue, bias, bias}, rewriter);
            Value reluValue = cloneBody(reluOp.getOperation()->getRegion(0).front(), {biasSum, biasSum}, rewriter);

            /** update Y[i, j] */
            rewriter.create<memref::StoreOp>(loc, reluValue, Y, ValueRange{i, j});

            //erase the original ops
            rewriter.eraseOp(gemmOp);
            rewriter.eraseOp(biasOp);
            rewriter.eraseOp(reluOp);

            //Return
            return true;
        }
        
        LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override
        {
            //collect ops
            GBROps gbrOps = collectOps(op);
            if(!gbrOps.gemmOp || !gbrOps.biasOp || !gbrOps.reluOp)
            {
                return failure();
            }

            //validiate ops
            bool isValid = isMatchedOps(gbrOps);
            if(!isValid)
            {
                return failure();
            }

            //rewrite
            bool isRewritten= rewriteOps(gbrOps, rewriter);
            if(!isRewritten)
            {
                return failure();
            }

            //return
            return success();
        }

    };

    //pass
    struct ToyFusionPass : PassWrapper<ToyFusionPass, OperationPass<ModuleOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyFusionPass);
        StringRef getArgument() const override
        {
            return "toy-mlp-fusion";
        }

        StringRef getDescription() const override
        {
            return "fusion a mlp";
        }

        void runOnOperation() override
        {
            //init
            RewritePatternSet patterns(&getContext());
            patterns.add<toy_mlp_fusion::ToyGEMMBiasReluFusionPattern>(&getContext());
            FrozenRewritePatternSet frozen(std::move(patterns));

            //apply
            if(failed(applyPatternsGreedily(getOperation(), frozen)))
            {
                signalPassFailure();
            }

        }
 
    };
}

namespace attention_fusion
{
    //
    struct QKScaleOps
    {
        linalg::GenericOp gemmOp;
        linalg::GenericOp scaleOp;
    };

    struct ExpSumOps
    {
        linalg::GenericOp expOp;
        linalg::GenericOp rowSumOp;
    };

    struct NormalizeGemmOps
    {
        linalg::GenericOp normalizeOp;
        linalg::MatmulOp gemmOp;
    };

    //three patterns
    /** QKScale Pattern */
    struct QKScalePattern : public OpRewritePattern<linalg::GenericOp>
    {
        using OpRewritePattern::OpRewritePattern;
        
        LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override
        {
            //collect
            
            //validate

            //rewrite

            //return
            return success();
        }
        
    };

    //pass
    struct AttentionFusionPass : public PassWrapper<AttentionFusionPass, OperationPass<func::FuncOp>>
    {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttentionFusionPass);

        StringRef getArgument() const override
        {
            return "toy-attention-fusion";
        }

        StringRef getDescription() const override
        {
            return "Fuse attention operations";
        }

        void runOnOperation() override
        {
            //init
            RewritePatternSet patterns(&getContext());
            patterns.add<attention_fusion::QKScalePattern>(&getContext());
            FrozenRewritePatternSet frozen(std::move(patterns));

            //apply
            if(failed(applyPatternsGreedily(getOperation(), frozen)))
            {
                signalPassFailure();
            }

        }

    };
    
}

//Register
void registerMyPass()
{
    PassRegistration<PrintOpsPass>();
    PassRegistration<FunctionStatsPass>();
    PassRegistration<LICMPass>();
    PassRegistration<DCEPass>();
    PassRegistration<CFPass>();
    PassRegistration<CSEPass>();
    PassRegistration<ICPass>();
    PassRegistration<ICPatternPass>();
    PassRegistration<LinalgAnalysisPass>();
    PassRegistration<FusionPass>();
    PassRegistration<TilingPass>();
    PassRegistration<ToyPromoteAccumulatorPass>();

    //gemm, mlp, transformer
    PassRegistration<toy_shared_tile::ToySharedTilePass>();
    PassRegistration<toy_mlp_fusion::ToyFusionPass>();
    PassRegistration<attention_fusion::AttentionFusionPass>();
}
