using ExprRules

# Count the number of nodes in an expression tree.
function count_nodes(tree, sum = 0)
    sum += 1
    for c in tree.children
        sum += count_nodes(c)
    end
    return sum
end

# Prune a tree until the cost function is above a certain value
function prune_unused_nodes(tree::RuleNode, grammar::Grammar, f::Function, threshold::Float64, leaf_type, ignore_terminals = [])
    pruned_tree  = tree
    while true
        leaves = get_leaves_of_type(pruned_tree, grammar, leaf_type, ignore_terminals)
        best_subtree, best_subtree_f = nothing, Inf
        for l in leaves
            new_tree = prune(pruned_tree, l, grammar)
            new_f = f(new_tree, grammar)
            if new_f < best_subtree_f
                best_subtree, best_subtree_f = new_tree, new_f
            end
        end
        if best_subtree_f < threshold && pruned_tree != best_subtree
            pruned_tree = best_subtree
        else
            break
        end
    end
    deepcopy(pruned_tree)
end

function get_leaves_of_type(tree, grammar, type, ignore_terminals = [])
    leaves = []
    if all(return_type(grammar, c) in ignore_terminals for c in tree.children) && return_type(grammar, tree) == type
        push!(leaves, tree)
    end
    for c in tree.children
        push!(leaves, get_leaves_of_type(c, grammar, type, ignore_terminals)...)
    end
    leaves
end

function prune(tree::RuleNode, leaf, grammar)
    tree_cp = deepcopy(tree)
    prune!(tree_cp, leaf, grammar)
    tree_cp
end

prune!(tree::RuleNode, leaf, grammar) = prune!([tree], leaf, grammar)

function prune!(lineage::Array{RuleNode}, leaf, grammar)
    if lineage[end] == leaf
        # Move up the lineage until a binary operator is found
        for i in length(lineage)-1:-1:1
            children = lineage[i].children[[c != lineage[i+1] for c in lineage[i].children]]
            if length(children) == 1 && return_type(grammar, children[1]) == return_type(grammar, lineage[i])
                replace_tree!(lineage[i], i==1 ? nothing : lineage[i-1], lineage[i+1])
                return true
            end
        end
        return false
    end
    # Otherwise, true to prune the children
    for c in lineage[end].children
        prune!([lineage..., c], leaf, grammar) && return true
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