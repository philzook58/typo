// use std::option;
use std::marker::PhantomData;
use std::ops::Add;
use std::mem;
const MINE : Option<i64> = None;
// https://github.com/gtestault/primitive-recursive-functions

struct Z {}
struct S<T>(PhantomData<T>);
type T1 = S<Z>;
struct True {}
struct False {}

trait ReflectBool {
    const val : bool; // maybe this should be a function that takles PhantomData and called reflectBool
}

impl ReflectBool for True{
    const val : bool  = true;
}

impl ReflectBool for False{
    const val : bool = false;
}

/*
suggestion 4.

impl ReflectBool for Either<A,B> where A : ReflectBool, B: ReflectBool {
    reflectBool(x : Either<A,B>) -> bool {
        match x {
            Left  y => reflectBool(y)
            Right y => reflectBool(y)
        }
        
    }
}

*/

// 1. unsafe coerce
// 2. require seperate functions. syntactically copy f. Turn reifyBool into macro. That's not so bad. We can do this because Bool is a finite universe. We don't really need forall s since our intention is that it only contains a finite universe of stuff
// 3. Fake higher rank functions somehow using a typeclass wrapper
// 4. Require an Either of all possible values. Implement reflection for this either via matching on left right tags.
fn reifyBool<F, G, W>( x : bool, f : F, g : G) -> W where
    F : FnOnce(PhantomData<True>) -> W, // has to take an Either of the different proxies?
    G : FnOnce(PhantomData<False>) -> W {
        if x {
            let q : PhantomData<True> = PhantomData;
            f(q)
        }
        else{
            let q : PhantomData<False> = PhantomData;
            g(q)
        }
    }
// Yea. I feel like this is kind of close to a GADT.

fn getVal<R>(x : PhantomData<R>) -> bool where R : ReflectBool  {  R::val    }
fn example() -> bool {reifyBool(true, getVal, getVal)} // this puts true into the type system and then back out
// if you want to use different functions for f and g, I feel like that is your perogative.


trait Not{
    type Res;
}
impl Not for True {
    type Res = False;
}

impl Not for False {
    type Res = True;
}
trait Nand<T> {
    type Res;
}
/*
impl Nand<False> for True{
    type Res = True;
}*/
impl<T> Nand<T> for True where T : Not {
    type Res = <T as Not>::Res;
} 

//let y : () = 

// HKT
//https://gist.github.com/CMCDragonkai/a5638f50c87d49f815b8
// Interesint. a HKT is somoething that can have interior exchanged.

// Equality is similar...
/*
trait Eq<A>  { // requires HKT
    type T; // context can change
    fn HKT<U,C> 

}


*/


struct OptionTag {}

trait Taggable {
    type Tag;
}

impl<T> Taggable for Option<T>{
    type Tag = OptionTag;
}

fn myid<A>(x : A) -> A {
    x
} 

fn mysnd<A,B>((x,y): (A,B)) -> B{
    y
}

struct V2<T> {
    x : T,
    y : T
}

impl<T> Add for V2<T> where T : Add<Output = T> {
    type Output = V2<T>;
    fn add(self, other : V2<T>) -> V2<T> {
        V2 { 
            x : self.x + other.x,
            y : self.y + other.y}
    }
}

// Make a compile time lisp.
// PrimOps have to be traits
// Make a macro to turn legit lisp expressions ()
// quote
// use typelevel num 
// using refl package, one could make a gcase macro
// to make casing on gadts easier.
struct Cons<A,B> {h: PhantomData<A>, t: PhantomData<B> }
struct Nil {}
type t3 = Cons<Z , Nil>;
// struct Apply<A,B> {f: PhantomData<A>, x: PhantomData<B> }
// struct Lam<V,B> {var : PhantomData<V>, body : PhantomData<B>}
type t4 = (bool,bool);
type KV<K,V> = (K,V);

trait Lookup<N> {
    type T;
}

impl<A,B> Lookup<Z> for Cons<A,B>{
    type T = A;
}

impl<N,A,B> Lookup<S<N>> for Cons<A,B> where B : Lookup<N> {
    type T = B::T;
}


//impl<K> Lookup<K> for (( , ), )
// kind of the same thing as HKT
/*
trait Lam<V, A> {
    type Output;
}
struct Id {};

impl<V,A> Lam<V,A> for Id {
    type Output = A;
}
*/
/*
struct Id {}
struct Snd {}
struct Fst {}
trait Apply<X> {
    type Output;
}
impl<X> Apply<X> for Id{
    type Output = X;
}

impl<X,Y> Apply<(X,Y)> for Snd{
    type Output = Y;
}

impl<X,Y> Apply<(X,Y)> for Fst{
    type Output = X;
}
*/
struct Id<X>(PhantomData<X>);
struct Fst<X>(PhantomData<X>);
trait Apply {
    type Output;
}

impl<X> Apply for Id<X> {
    type Output = X;
}

impl<X,Y> Apply for Fst<(X,Y)> {
    type Output = X;
}
// trait Eval // trait Apply?
#[derive(Clone, Debug)]
enum LamExpr {
    Lam( Box<LamExpr>),
    App( Box<LamExpr>,  Box<LamExpr>),
    Var( usize),
    Lit(i64) // Pair, Fst, Snd
}
use LamExpr::*;

/*
split enum out may
struct Lam<U>( )

or... 
reflection into typeclass via 
class Reflect s
    yada yada
    yada yada
    yada yada
use macro to make fresh reflection class every time.

PhantomData<True>
trait Reflect where
    type Output;
    val : Output;
impl Reflect for True where
    type Output = bool;
    val = true;
impl Reflect for False{
    type Output = bool;
    val = false;
    }
-- reflecting ordinary types
-- some kind of dyanmic typing might pull this off?
-- Any
impl Reflect for Vec<U> where
    type Output = VecR<U::Output>; 
    val = VecR(val);

reflection can perform something very singleton like.
typeclasses cna perform higher rank functionality.

(Reflect fi) (Reflect fo) (Reflect fs) => reifyWith x match x => 

reifyWith ::  MyGADT s' -> (Reflect s, Reflect fs) => s -> fs) (Reflect s) (Reflect s) -> f s'
reifyWith 

reifyWith 
refiyWithVecMyGADT Reflect s => MyGADT s' -> Vec<s::out> -> Vec<s::out> -> Vec<s> -> Vec<s::out>
matchish Reflect s => 

reifyWith GADT s -> (Reflect s s' => f s' -> f s') -> f s
Reflect<Int> s

Reify
Eq<Int> for U{

}

Eq<>

typclass equality

Eq<U> {
    cast :: Self -> U
    uncast :: U -> Self
}

Eq<i64> for i64{
      ?leibnitz?
      cast = id
      uncast = id
}

s : Eq<Int>
s' : Eq<Bool>

(Type -> GARY) -> 

refl : Eq<A,A>
withRefl 


App<Vec<Type>,A>
impl Eval for App<Vec<Type>,A> where A : Eval{
    type Output = Vec<A::Output>
}
impl Eval for i64{
    type Output = i64;
}

impl Abstract for Vec<A> {
    type Ouput = App<Vec<Type>,A::Abstract
}

impl EvalOrAbstract<A> {

}

Unify f<Int> (App<Vec<Type>,>) where
 -> App<F,Int> -> App<F,Bool> -> 
 reify( c1 : impl Eval<App<F,Int>>, c2 : impl ) where fi = App<F,Int> as Eval :: Output 
refiy( c1 : fi, )

App

we might need something more like the coq case syntax giving the explanation on how to build the type of the result.
This is only acceptable with a tactic language.

desctrcut<F> :: GADT s -> Eval App<F,Int> -> Eval App<F,Bool> -> Eval App<F,s>
destructEq :: Eq a b -> Eval App2<F,A,A> -> Eval App2<F,A,B>

struct Lam<A,B>
type MyExpr<A> = Lam<A, Vec<A>> // sort of a typelevel HOAS
lam!(fred = \a b -> b) -> type Fred0<A,B> = Fred<>
need to curry everythign? 
lam!(fred = \a b -> Vec a)   type Fred<A,B> = Vec<A>
pat!(fred a b = )
lam!(fred = \a b c -> ) ==> type Fred<A,B,C> = 

impl Apply Fred<Type,Type,Type>




Abtract 
HKT = Abstract

fn<U>( )

f s -> f s -> f s ->


R -> R -> R
   U ->  U -> U -> U
require GADT destructor for every pair of GADT to destruct and context to put it in?
destruct
maybe via reflection? reflect type variables into 
desugar into lambda   |x y z| =>  
a higher order macro. Can we do that?

smart constructors
fn lam(a : LamExpr<U>) -> LamExpr<Arr<U,V>>

returning a type that only implements a trait is a way of exitentializing.

*/
// https://gist.github.com/kyleheadley/af149f7452c8163667e2b047743a9e44
// typevel lambda calc
fn eval(mut env : Vec<LamExpr>, term : LamExpr) -> LamExpr {

    match term{
        LamExpr::App(l, v) => {
            let env2 = env.clone();
            let v2 = eval( env2, *v);
            env.push(v2);
            eval( env  , *l)
            },
        LamExpr::Lam(b) => eval(env, *b),
        LamExpr::Var(i) => {
            if env.len() > i{
                env[i].clone()
            }
            else {
                Var(i)
            }            
        },
        Lit(x) => Lit(x)
    }
}

fn app(f : LamExpr, x : LamExpr) -> LamExpr{
    LamExpr::App(Box::new(f), Box::new(x))
}
fn lam(b : LamExpr) -> LamExpr{
    LamExpr::Lam(Box::new(b))
}

fn id() -> LamExpr {
   LamExpr::Lam(Box::new(Var(0)))
}

fn pair(a : LamExpr, b : LamExpr) -> LamExpr{
    lam(app(app(Var(0), a), b))
}
fn fst() -> LamExpr{
    lam(lam( Var(1)))
}

fn snd() -> LamExpr{
    lam(lam( Var(0)))
}

struct TypeTag {}

struct List<A> {
    head : A, 
    tail : Box<List<A>>
}

trait ApplyHKT<U> {
    type Output;
}

trait AbstractHKT {
    type Output;
}
// splitting the suggested HKT typeclass into two method.
// may want tag types the be macro generated to be units.
impl<U> AbstractHKT for List<U>{
    type Output = List<TypeTag>;
}

impl<U> ApplyHKT<U> for List<TypeTag>{
    type Output = List<U>;
}
enum Lit<U> {
    LitInt(i64, PhantomData<U>),
    LitBool(bool, PhantomData<U>)
}
/*
trait GadtTest : HKT<i64> HKT<bool> {
    fn litint(i64) -> Self::HKT<i64>::T;
    fn litbool(bool) -> Self::HKT<bool>::T;
}
*/

struct Lam<B> (PhantomData<B>);

type T5 = Lam<List<Z>>; // The de bruijn indexed List type

trait ApplyDB<Env> {
    type T;
}

trait Eval<Env> {

}
/*
struct Apply2<L,X> {lam : PhantomData<L> , var : PhantomData<X>  }

impl<Env> Eval where {

} 
impl<B, Env> ApplyDB<Env> Lam<B> where Apply<Cons<Arg, Env>>  {

}

*/
// sized Vec?
// I mean I think rust has sized arrayc built in, but still.

trait App1<A> {
    type T;
}



enum Gadt<A> {
    TInt(PhantomData<A>),
    TBool(PhantomData<A>)
    
}
// ould macro desgar TInt to TIntINTERNALDONOTUSE + a random string
// then build the specialzied constructors that 
fn TBool2() -> Gadt<bool>{
    Gadt::TBool(PhantomData)
}

fn tInt() -> Gadt<i64>{
    Gadt::TInt(PhantomData)
}

// every use site of gmatch reuqires an instance of App1 connected to a function symbol.
struct F1 {}
impl<A> App1<A> for F1{
    type T = Vec<Vec<A>>;
}
// type F1 = HKT!( Vec<Vec<0,1,2>><2>  )
// match 
// presumably a match can be desugared by a macro. Build the above struct and instance, and turn the cases into 

// case1 will often be a closure type.
fn gmatch<F,S>(x : Gadt<S>, case1 : <F as App1<bool>>::T   , case2 : <F as App1<i64>>::T ) -> <F as App1<S>>::T  where 
         F : App1<bool>,
         F : App1<i64>,
         F : App1<S>
         {
    match x{
        Gadt::TBool(_) => unsafe{mem::transmute_copy(&case1)}, // I guess this is horrifying
        Gadt::TInt(_) => unsafe{mem::transmute_copy(&case2)}
    }
}

/*
enum Gadt2<A> {
    TIntINTERNAL(PhantomData<A>),
    TBoolINTERNAL(PhantomData<A>)
    
}
fn TBool2() -> Gadt<bool>{
    Gadt2::TBool(PhantomData)
}

fn tInt() -> Gadt<i64>{
    Gadt2::TInt(PhantomData)
}
*/

/*
a more satifying possiblity. Make an eliminator typeclass
This allows us to 
The wrong branches are not type safe, but these should never get touched.
We could also perhaps return a option, and then unsafe unwrap the option.
*/

trait GadtElim {
    type Inner;
fn gadtElim<F>(&self, case1 : <F as App1<bool>>::T   , case2 : <F as App1<i64>>::T ) -> <F as App1<Self::Inner>>::T  where 
         F : App1<bool>,
         F : App1<i64>,
         F : App1<Self::Inner>;
         

}


impl GadtElim for Gadt<i64> {
    type Inner = i64;
    fn gadtElim<F>(&self, case1 : <F as App1<bool>>::T   , case2 : <F as App1<i64>>::T ) -> <F as App1<Self::Inner>>::T  where 
         F : App1<bool>,
         F : App1<i64>,
         F : App1<Self::Inner>{
        match self{
            Gadt::TInt(PhantomData) => case2,
            Gadt::TBool(PhantomData) => panic!("Somehow TBool has type Gadt<i64>")// Will never be reached though. god willing
        }
    }
}
impl GadtElim for Gadt<bool> {
    type Inner = bool;
    fn gadtElim<F>(&self, case1 : <F as App1<bool>>::T   , case2 : <F as App1<i64>>::T ) -> <F as App1<Self::Inner>>::T  where 
         F : App1<bool>,
         F : App1<i64>,
         F : App1<Self::Inner>{
        match self{
            Gadt::TInt(PhantomData) => panic!("Somehow TInt has type Gadt<bool>"),
            Gadt::TBool(PhantomData) => case1 // Will never be reached though. god willing
        }
    }
}


struct Eq<A,B>(PhantomData<A>, PhantomData<B>); // don't even bother having a constructor
// Mauybe we should call the whole thing Refl?

trait App2<A,B> {
    type T;
}
// family!(F<A,B> = Vec<yadayaydayd>) => struct F {} + impl<A,A> AppN<A,B> for F {type T = Vec yadaya}
// altenrative
fn matchEq<F,A,B>(x : Eq<A,B>, case1 : <F as App2<A,A>>::T) -> <F as App2<A,B>>::T where
 F : App2<A,B>,
 F : App2<A,A>
 {
    unsafe{mem::transmute_copy(&case1)}
}

// reflect!( enum Yada = { , , , }  )
// singleton!()
// 


// This is pretty similar to the implementation of refl.
// He's got some stuff we hsould also do. ?Sized annotations
// and he has mem forget the original reference.
// https://www.reddit.com/r/rust/comments/9neop8/are_there_any_projects_trying_to_emulate/

// That our trusted codebase is built by macro is pretty horrifying.

fn main() {

    println!("Hello, world!");
    dbg!(MINE);
    dbg!(mysnd((1,"Hi")));


    dbg!(eval(Vec::new(),  app(id(),Lit(1))  ));
    dbg!(eval(Vec::new(),  id()));
    dbg!(eval(Vec::new(),  app(fst(), pair(Lit(0),Lit(1)))     ));
    dbg!(example());

}
