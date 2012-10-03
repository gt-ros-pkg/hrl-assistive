/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Robert Bosch LLC.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Robert Bosch nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************/

/**
 * Represents a Tree of Objects of generic type T. The Tree is represented as
 * a single rootElement which points to a Array of children. There is
 * no restriction on the number of children that a particular node may have.
 * This Tree provides a method to serialize the Tree into a List by doing a
 * pre-order traversal. It has several methods to allow easy updation of Nodes
 * in the Tree.
 */
ros.Tree =  Class.extend(
/** @lends ros.Tree# */
{

	init : function(){
		this.rootElement = null;
	},

 
/**
 * Return the root Node of the tree.
 * @return the root element.
 */
	getRootElement : function() {
		return this.rootElement;
	},
 
/**
 * Set the root Element for the tree.
 * @param rootElement the root element to set.
 */
	setRootElement : function(rootElement) {
		this.rootElement = rootElement;
	},
     
/**
 * Returns the ros.Tree as a List of ros.TreeNode objects. The elements of the
 * List are generated from a pre-order traversal of the tree.
 * @return a Array.
 */
	toList : function() {
		var list = new Array();
		walk(this.rootElement, list);
		return list;
	},
     
/**
 * Returns a String representation of the Tree. The elements are generated
 * from a pre-order traversal of the Tree.
 * @return the String representation of the Tree.
 */
	toString : function() {
		return toList().toString();
	},
     
/**
 * Walks the Tree in pre-order style. This is a recursive method, and is
 * called from the toList() method with the root element as the first
 * argument. It appends to the second argument, which is passed by reference     * as it recurses down the tree.
 * @param element the starting element.
 * @param list the output of the walk.
 */
	walk : function(element, list) {
		list.push(element);
		var children = element.getChildren();
		for (var i in children) {
			walk(children[i], list);
		}
	},

/**
 * Represents a node of the ros.Tree class. The ros.TreeNode is also a container, and
 * can be thought of as instrumentation to determine the location of the type T
 * in the ros.Tree.
 */
	TreeNode : function (data) {
		this.data = data;
		this.children = [];
	},
	
/**
 * Return the children of ros.TreeNode. The ros.Tree is represented by a single
 * root ros.TreeNode whose children are represented by a Array. Each of
 * these ros.TreeNode elements in the List can have children. The getChildren()
 * method will return the children of a ros.TreeNode.
 * @return the children of ros.TreeNode
 */
	getChildren : function() {
		return this.children;
	},
 
/**
 * Sets the children of a ros.TreeNode object. See docs for getChildren() for
 * more information.
 * @param children the Array to set.
 */
	setChildren : function(children) {
		this.children = children;
	},
 
/**
 * Returns the number of immediate children of this ros.TreeNode.
 * @return the number of immediate children.
 */
	getNumberOfChildren : function() {
		return children.length;
	},
     
/**
 * Adds a child to the list of children for this ros.TreeNode. The addition of
 * the first child will create a new Array.
 * @param child a ros.TreeNode object to set.
 */
	addChild : function(child) {
		children.push(child);
	},
     
/**
 * Inserts a ros.TreeNode at the specified position in the child list. 
 * @param index the position to insert at.
 * @param child the ros.TreeNode object to insert.
 * @throws IndexOutOfBoundsException if thrown.
 */
	insertChildAt : function(index, child) {
		if (index == getNumberOfChildren()) {
			// this is really an append
			addChild(child);
			return;
		} else {
			children.splice(index, 0, child);
		}
	},
     
/**
 * Remove the ros.TreeNode element at index index of the Array.
 * @param index the index of the element to delete.
 * @throws IndexOutOfBoundsException if thrown.
 */
	removeChildAt : function(index) {
		children.splice(index,1);
	},
 
	getData : function() {
		return this.data;
	},
 
	setData : function(data) {
		this.data = data;
	},
     
	toString : function() {
		var str = new String();
		str.concat("{",getData().toString(),",[");
		var i = 0;
		for (var c in children) {
			if (i > 0) {
				str.concat(",");
			}
			str.concat(children[c].getData().toString());
			i++;
		}
		str.concat("]","}");
		return str;
	},
});



