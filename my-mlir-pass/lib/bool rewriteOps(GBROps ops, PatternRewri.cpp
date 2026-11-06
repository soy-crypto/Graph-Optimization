bool rewriteOps(GBROps ops, PatternRewriter &rewriter) const
{
    // isMatchedOps() has already checked matching and fusion safety.
    auto gemmOp = ops.gemmOp;
    auto biasOp = ops.biasOp;
    auto reluOp = ops.reluOp;

    Value X  = gemmOp.getDpsInputs()[0];
    Value W1 = gemmOp.getDpsInputs()[1];
    Value T1 = gemmOp.getDpsInits()[0];
    Value B1 = biasOp.getDpsInputs()[1];
    Value Y  = reluOp.getDpsInits()[0];

    Location loc = gemmOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(gemmOp);

    Value c0  = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c64 = rewriter.create<arith::ConstantIndexOp>(loc, 64);
    Value c1  = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Create the row loop.
    auto iLoop = rewriter.create<scf::ForOp>(loc, c0, c64, c1);
    rewriter.setInsertionPointToStart(iLoop.getBody());
    Value i = iLoop.getInductionVar();

    // Create the column loop inside the row loop.
    auto jLoop = rewriter.create<scf::ForOp>(loc, c0, c64, c1);
    rewriter.setInsertionPointToStart(jLoop.getBody());
    Value j = jLoop.getInductionVar();

    Value initialAcc = rewriter.create<memref::LoadOp>(loc, T1, ValueRange{i, j});

    // Create the reduction loop with an accumulator.
    auto kLoop = rewriter.create<scf::ForOp>(loc, c0, c64, c1, ValueRange{initialAcc});

    Block *kBody = kLoop.getBody();
    if (!kBody->empty() && isa<scf::YieldOp>(kBody->back()))
        rewriter.eraseOp(&kBody->back());

    // Insert subsequent operations into the reduction-loop body.
    rewriter.setInsertionPointToStart(kBody);

    Value k = kLoop.getInductionVar();
    Value acc = kLoop.getRegionIterArgs()[0];
    Value x = rewriter.create<memref::LoadOp>(loc, X, ValueRange{i, k});
    Value w = rewriter.create<memref::LoadOp>(loc, W1, ValueRange{k, j});

    // Compiler calls cloneBody once here.
    // The generated arithmetic executes on every k iteration.
    Value nextAcc = cloneBody(gemmOp->getRegion(0).front(), ValueRange{x, w, acc}, rewriter);
    rewriter.create<scf::YieldOp>(loc, ValueRange{nextAcc});

    // Insert after the reduction, still inside the column loop.
    rewriter.setInsertionPointAfter(kLoop);
    Value sum = kLoop.getResult(0);
    Value bias = rewriter.create<memref::LoadOp>(loc, B1, ValueRange{j});

    // Second compiler call: generate bias addition.
    Value biased = cloneBody(biasOp->getRegion(0).front(),ValueRange{sum, bias, sum},rewriter);

    // Third compiler call: generate ReLU.
    Value activated = cloneBody(reluOp->getRegion(0).front(),ValueRange{biased, biased},rewriter);

    rewriter.create<memref::StoreOp>(loc, activated, Y, ValueRange{i, j});

    // Remove the original operations.
    rewriter.eraseOp(reluOp);
    rewriter.eraseOp(biasOp);
    rewriter.eraseOp(gemmOp);

    return true;
}