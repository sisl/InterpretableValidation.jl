# Count the number of nodes in an expression tree.
function count_nodes(tree, sum = 0)
    sum += 1
    for c in tree.children
        sum += count_nodes(c)
    end
    return sum
end

# Prune a tree until the cost function is above a certain value
function prune_unused_nodes(tree::RuleNode, grammar::Grammar, f::Function, threshold::Float64, leaf_type = :τ, binary_terminals_to_ingore = :C)
    pruned_tree  = tree
    while true
        leaves = get_leaves(pruned_tree, grammar, leaf_type)
        best_subtree, best_subtree_f = nothing, Inf
        for l in leaves
            new_tree = prune(pruned_tree, l, grammar, binary_terminals_to_ingore)
            new_f = f(new_tree, grammar)
            if new_f < best_subtree_f
                best_subtree, best_subtree_f = new_tree, new_f
            end
        end
        if best_subtree_f < threshold
            pruned_tree = best_subtree
        else
            break
        end
    end
    deepcopy(pruned_tree)
end

function get_leaves(tree, grammar, type = :τ)
    leaves = []
    if isempty(tree.children) && return_type(grammar, tree) == type
        push!(leaves, tree)
    end
    for c in tree.children
        push!(leaves, get_leaves(c, grammar, type)...)
    end
    leaves
end

function prune(tree::RuleNode, leaf, grammar, binary_terminals_to_ingore=:C)
    tree_cp = deepcopy(tree)
    prune!(tree_cp, leaf, grammar, binary_terminals_to_ingore)
    tree_cp
end

prune!(tree::RuleNode, leaf, grammar, binary_terminals_to_ingore=:C) = prune!([tree], leaf, grammar, binary_terminals_to_ingore)

function prune!(lineage::Array{RuleNode}, leaf, grammar, binary_terminals_to_ingore=:C)
    if lineage[end] == leaf
        # Move up the lineage until a binary operator is found
        for i in length(lineage)-1:-1:1
            ctypes = child_types(grammar, lineage[i].ind)
            if length(ctypes) == 2 && binary_terminals_to_ingore ∉ ctypes
                replace_tree!(lineage[i], i==1 ? nothing : lineage[i-1], lineage[i+1])
                return true
            end
        end
        return false
    end
    # Otherwise, true to prune the children
    for c in lineage[end].children
        prune!([lineage..., c], leaf, grammar, binary_terminals_to_ingore) && return true
    end
    false
end


# replace "tree" with one of its children (the one that is not child to remove)
# by replacing its spot in the parent.
# Assumes tree has exactly 2 children
function replace_tree!(tree, parent, child_to_remove)
    new_tree = filter(x->x!=child_to_remove, tree.children)[end]
    if isnothing(parent)
        tree.ind = new_tree.ind
        tree._val = new_tree._val
        while length(tree.children) > 0
            deleteat!(tree.children,1)
        end
        for c in new_tree.children
            push!(tree.children, c)
        end
    else
        tree_ind = findfirst(x->x==tree, parent.children)
        parent.children[tree_ind] = new_tree
    end
end

has_child(child, tree) = ~isempty(findall(x->x==child, tree.children))