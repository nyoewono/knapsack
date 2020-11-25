#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:50:23 2020

@author: nathanaelyoewono
"""

import numpy as np

class KnapSack:
    
    """
    * This will only handle discrete cases
    * Required library to run this program:
        - numpy
    """
    
    def __init__(self, sizes, values, max_size):
        """
        Argument types:
        * sizes = list
        * values = list
        * max_size = int
        """
        self.sizes = sizes
        self.values = values
        self.max_size = max_size
    
    def _initialize_matrix(self):
        """Initialise the possible sacks with sack object in each index"""
        self.sacks = np.zeros((len(self.values), self.max_size+1), dtype='object')
        self._initialize_sacks()
    
    def _initialize_sacks(self):
        """Initialise to put in each index in the matrix a sack object for retracing"""
        for i in range(len(self.values)):
            for j in range(self.max_size+1):
                set_sack = Sack(i)
                self.sacks[i, j] = set_sack
                
    def _backward_dynamic(self):
        """Do a backward dynamic programming"""
        # iterate from the end of the item
        for i in range(len(self.values)-1, 0, -1):
            
            #print('Allocate item:', i+1)
            
            # set the value function at time N
            item_size = self.sizes[i]
            item_value = self.values[i]
            
            # iterate in each sacks for item i (this is the current size)
            for j in range(self.max_size+1):
                self._iter_function(item_size, j, item_value, i)
        
        #print('Allocate item:', 1)
        # first item, use all available weights
        first_item_alloc = self._get_all_alloc(self.sizes[0], self.max_size)
        
        self._iter_function(self.sizes[0], self.max_size, self.values[0], 0)
        
    
    def _get_all_alloc(self, item_size, cur_size):
        """Get the list of possible item allocation in the sack"""
        floor_alloc = int(cur_size/item_size)
        lst_alloc = list(range(0, floor_alloc+1))
        return lst_alloc
    
    def _max_alloc(self, lst_alloc, item_val, item, cur_size, item_size):
        """Return the maximal allocation of an item based on the accumulated value"""
        
        # this part works
        if item==len(self.values)-1:
            lst_max_val = np.array(lst_alloc.copy())
            lst_max_val = lst_max_val*np.array(item_val)
            max_alloc = np.argmax(lst_max_val)
            self.sacks[item, cur_size].update_sack(item, max_alloc, item_val, item_size)
        else:
            lst_sacks = []
            # iterate for each possible item allocation
            for each_alloc in lst_alloc:
                
                # generate new sack object and only take the one with max value
                cad_sack = Sack(item)
                
                # update the object with the next accumulated sacks
                cad_sack.update_sack(item, each_alloc, item_val, item_size)
                update_cur_size = cur_size-(each_alloc*item_size)
                cad_sack.combine_sacks(self.sacks[item+1, update_cur_size])
                
                # for debugging
                #print('Updated sack: ')
                #print('Current value:', cad_sack.cur_value)
                #print('Current items:', cad_sack.items)
                
                # save the new cadidate sack
                lst_sacks.append(cad_sack)
            
            # get the highest accumulated sack
            lst_max_sacks = np.array([i.cur_value for i in lst_sacks])
            max_sacks = lst_sacks[np.argmax(lst_max_sacks)]
            self.sacks[item, cur_size] = max_sacks
    
    def _iter_function(self, item_size, cur_size, item_value, item):
        lst_alloc = self._get_all_alloc(item_size, cur_size)
        self._max_alloc(lst_alloc, item_value, item, cur_size, item_size)
    
    def solve(self):
        """return the most optimal sack"""
        self._initialize_matrix()
        self._initialize_sacks()
        self._backward_dynamic()
        self._display_sack(self.sacks[0, -1])
        return self.sacks[0, -1]
    
    def _display_sack(self, sack):
        print('-'*17)
        print('*Best Allocation*\n')
        for each in sack.items:
            print(f'Item {each+1} : {sack.items[each]}')
        print('\n')
        print('Value:', sack.cur_value)
        print('-'*17)
        

class Sack:
    """
    Sack object to help store and retrace the most optimal allocation
    """
    
    def __init__(self, init_item):
        self.items = {init_item:0}
        self.cur_value = 0
    
    def update_sack(self, item, alloc, value, size):
        self.items[item] = alloc
        self.cur_value += value*alloc
    
    def combine_sacks(self, sack_b):
        self.cur_value += sack_b.cur_value
        self.items.update(sack_b.items)
    