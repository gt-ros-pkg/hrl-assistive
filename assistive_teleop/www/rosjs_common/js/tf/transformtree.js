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
 * Represents a TransformTree of Objects of generic type. The TransformTree is represented as
 * a single rootNode which points to a Array of children. There is
 * no restriction on the number of children that a particular node may have.
 * This TransformTree provides a method to serialize the TransformTree into a List by doing a
 * pre-order traversal. It has several methods to allow easy updation of Nodes
 * in the TransformTree.
 */
ros.tf.TransformTree = Class.extend(
/** @lends ros.tf.TransformTree# */
{
	init: function(){
		this.rootNode = null;
	},

 
/**
 * Return the root Node of the tree.
 * @return the root element.
 */
	getRootNode : function() {
		return this.rootNode;
	},
 
/**
 * Set the root Element for the tree.
 * @param rootNode the root element to set.
 */
	setRootNode : function(rootNode) {
		this.rootNode = rootNode;
	},
     
/**
 * Returns the ros.tf.TransformTree as a List of ros.tf.TransformTreeNode objects. The elements of the
 * List are generated from a pre-order traversal of the tree.
 * @return a Array.
 */
	toList : function() {
		var list = new Array();
		this.walk(this.rootNode, list);
		return list;
	},
     
/**
 * Returns a String representation of the TransformTree. The elements are generated
 * from a pre-order traversal of the TransformTree.
 * @return the String representation of the TransformTree.
 */
	toString : function() {
		return toList().toString();
	},
     
/**
 * Walks the TransformTree in pre-order style. This is a recursive method, and is
 * called from the toList() method with the root element as the first
 * argument. It appends to the second argument, which is passed by reference     * as it recurses down the tree.
 * @param element the starting element.
 * @param list the output of the walk.
 */
	walk : function(element, list) {
		list.push(element);
		var children = element.getChildren();
		for (var i in children) {
			this.walk(children[i], list);
		}
	},

});






/**
 * Represents a node of the ros.tf.TransformTree class. The ros.tf.TransformTreeNode is also a container, and
 * can be thought of as instrumentation to determine the location of the type T
 * in the ros.tf.TransformTree.
 */


ros.tf.TransformTreeNode  = Class.extend(
/** @lends ros.tf.TransformTreeNode# */
{
		
	init: function (frame_id) {
		this.frame_id = frame_id;
		this.parent = null;
		this.transforms = [];
		this.children = [];
	},

/**
 * Return the children of ros.tf.TransformTreeNode. The ros.tf.TransformTree is represented by a single
 * root ros.tf.TransformTreeNode whose children are represented by a Array. Each of
 * these ros.tf.TransformTreeNode elements in the List can have children. The getChildren()
 * method will return the children of a ros.tf.TransformTreeNode.
 * @return the children of ros.tf.TransformTreeNode
 */
	getChildren : function() {
		return this.children;
	},
 
/**
 * Returns the child with a specific frame id
 * @return the child of ros.tf.TransformTreeNode or null if no child with this frame id
 */
	getChild : function(frame_id) {
		for (var c in this.children) {
			var child = this.children[c];
			if(child.frame_id == frame_id) {
				return child;
			}
		}
		return null;
	},

/**
 * Returns the child with a specific frame id
 * @return the child of ros.tf.TransformTreeNode or null if no child with this frame id
 */
	getChildID : function(frame_id) {
		for (var c in this.children) {
			var child = this.children[c];
			if(child.frame_id == frame_id) {
				return c;
			}
		}
		return -1;
	},

/**
 * Sets the children of a ros.tf.TransformTreeNode object. See docs for getChildren() for
 * more information.
 * @param children the Array to set.
 */
	setChildren : function(children) {
		this.children = children;
	},
 
/**
 * Returns the number of immediate children of this ros.tf.TransformTreeNode.
 * @return the number of immediate children.
 */
	getNumberOfChildren : function() {
		return children.length;
	},
     
/**
 * Adds a child to the list of children for this ros.tf.TransformTreeNode. The addition of
 * the first child will create a new Array.
 * @param child a ros.tf.TransformTreeNode object to set.
 */
	addChild : function(child) {
		children.push(child);
	},
     
/**
 * Inserts a ros.tf.TransformTreeNode at the specified position in the child list. 
 * @param index the position to insert at.
 * @param child the ros.tf.TransformTreeNode object to insert.
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
 * Remove the ros.tf.TransformTreeNode element at index index of the Array.
 * @param index the index of the element to delete.
 * @throws IndexOutOfBoundsException if thrown.
 */
	removeChildAt : function(index) {
		children.splice(index,1);
	},
 
	getFrame : function() {
		return this.frame_id;
	},
 
	setFrame : function(frame_id) {
		this.frame_id = frame_id;
	},
     
	toString : function() {
		var str = new String();
		str.concat("{",getFrame().toString(),",[");
		var i = 0;
		for (var c in children) {
			if (i > 0) {
				str.concat(",");
			}
			str.concat(children[c].getFrame().toString());
			i++;
		}
		str.concat("]","}");
		return str;
	},
	
});



