example programs from the book scientific and engineering programming in c++ an introduction with advanced techniques and examples addison wesley 1994 c copyright international business machines corporation 1994 all rights reserved see readme file for further details ifndef acme130gctch define acme130gctch include examples ch9 voltagesupply h include examples ch9 gpibcontroller h include examples ch9 gpibinstrument h include examples ch12 gpibcomponent_tc h class acme130_gctc public virtual voltagesupply public virtual gpibinstrument public gpibcomponent_tc public virtual kind kind const return acme130component rest like acme130_vs_gi_gc acme130_gctc gpibcontroller& controller int gpib_address voltagesupply interface virtual void set float volts virtual float minimum const virtual float maximum const gpibinstrument interface virtual void send const char virtual void send float f virtual float receive private gpibcontroller& my_controller int my_gpib_address inline float acme130_gctc minimum const return 0.0 float acme130_gctc maximum const return 10.0 inline void acme130_gctc send float value my_controller send my_gpib_address value inline void acme130_gctc send const char cmd my_controller send my_gpib_address cmd inline float acme130_gctc receive return my_controller receive my_gpib_address inline acme130_gctc acme130_gctc gpibcontroller& controller int gpib_address my_controller controller my_gpib_address gpib_address my_controller insert acme130_gctc gpib_address inline void acme130_gctc set float voltage if voltage maximum voltage minimum throw acme 130 voltage out of range send voltage endif
